import os
import gzip
import gc
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import multiprocessing as mp
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.cloud import storage


# def initialize_progen2_noeval(model_name):
#     '''
#     Initializes the ProGen2 model with the given name.
#     '''
    
#     # work around for google cloud vm
#     if not hasattr(model, "all_tied_weights_keys"):
#         model.all_tied_weights_keys = set()
#     if not hasattr(model, "_tied_weights_keys"):
#         model._tied_weights_keys = set()

#     # Define tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

#     return model, tokenizer

def initialize_progen2_noeval(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    from transformers.modeling_utils import PreTrainedModel

    _original_adjust = PreTrainedModel._adjust_tied_keys_with_tied_pointers

    def safe_adjust(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        if not hasattr(self, "_tied_weights_keys"):
            self._tied_weights_keys = {}
        return _original_adjust(self, *args, **kwargs)

    PreTrainedModel._adjust_tied_keys_with_tied_pointers = safe_adjust
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        device_map={"": "cpu"}
    )
    # Force ALL tensors/buffers to CPU
    model = model.to("cpu")

    # Also move any remaining meta tensors explicitly
    for name, param in list(model.named_parameters()):
        if param.device.type == "meta":
            print(f"Moving {name} from meta to cpu")
            param.data = torch.empty(param.shape, dtype=param.dtype, device="cpu")

    for name, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            print(f"Moving buffer {name} from meta to cpu")
            # find the module and replace the buffer
            parts = name.split(".")
            module = model
            for p in parts[:-1]:
                module = getattr(module, p)
            module.register_buffer(parts[-1], torch.zeros(buf.shape, dtype=buf.dtype, device="cpu"))
    
    # scale_attn is an attribute, not a registered buffer, so it escapes the above loop
    for module in model.modules():
        if hasattr(module, 'scale_attn') and isinstance(module.scale_attn, torch.Tensor):
            if module.scale_attn.device.type == 'meta':
                # print("Fixing scale_attn on meta device")
                shape = module.scale_attn.shape
                dtype = module.scale_attn.dtype
                module.scale_attn = torch.ones(shape, dtype=dtype, device='cpu')
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     trust_remote_code=True,
    #     low_cpu_mem_usage=False  # force weights onto cpu, not meta device
    # )
    # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    PreTrainedModel._adjust_tied_keys_with_tied_pointers = _original_adjust

    # Patch get_head_mask on BOTH the outer and inner model
    import types

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    # Patch outer ProGenForCausalLM
    if not hasattr(model, "get_head_mask"):
        model.get_head_mask = types.MethodType(get_head_mask, model)

    # Patch inner ProGenModel (model.transformer)
    if not hasattr(model.transformer, "get_head_mask"):
        model.transformer.get_head_mask = types.MethodType(get_head_mask, model.transformer)

    return model, tokenizer


def collect_log_prob_pg2(sequence, model, tokenizer):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_token_ids = [tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids]

    prompt1 = "1" + sequence
    prompt2 = "2" + sequence[::-1]

    input_ids1 = torch.tensor(tokenizer.encode(prompt1)).unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits1 = model(input_ids1).logits
    shift_logits1 = logits1[:, :-1, :]

    input_ids2 = torch.tensor(tokenizer.encode(prompt2)).unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits2 = model(input_ids2).logits
    shift_logits2 = logits2[:, :-1, :]
    shift_logits2 = shift_logits2[:, torch.arange(shift_logits2.size(1) - 1, -1, -1), :]

    input_ids = input_ids1[:, 1:]
    logits = (shift_logits1 + shift_logits2) / 2
    log_probs = F.log_softmax(logits, dim=-1)
    ref_log_probs = log_probs[0, torch.arange(input_ids.size(1)), input_ids[0]].unsqueeze(1)
    llr_matrix = (log_probs - ref_log_probs)[0][:, aa_token_ids]
    log_probs = log_probs[0][:, aa_token_ids]

    return np.array(log_probs), np.array(ref_log_probs), np.array(llr_matrix)


def listwise_ranking_loss(preds, targets):
    indices = targets.sort(descending=True).indices
    preds = torch.gather(preds, dim=-1, index=indices)
    cumsums = preds.exp().flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
    loss = torch.log(cumsums + 1e-10) - preds
    return loss.mean()


def FineTune_ProGen2_LORA(device, base_model, tokenizer, lora_config,
                          protein_seq, dom_seq, dom_pos, target_tensor, loss_fn,
                          lrate=1e-3, num_epochs=5, k=0.8, num_samples=20, print_info=True):

    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_token_ids = [tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids]

    model = get_peft_model(base_model, lora_config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lrate)

    seq_len = target_tensor.shape[0]
    all_indices = torch.randperm(seq_len)
    num_train = int(seq_len * k)
    train_indices = all_indices[:num_train]
    val_indices = all_indices[num_train:]

    train_losses = []
    val_losses = []

    vocab_dict = tokenizer.get_vocab()
    seq_list = list(protein_seq)
    inputs = tokenizer(protein_seq, return_tensors="pt").to(device)

    for epoch in range(num_epochs):
        model.train()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)
        wt_logits = torch.log_softmax(logits, dim=-1)
        residue_indices = torch.arange(len(protein_seq))
        seq_indices = [vocab_dict[aa] for aa in seq_list]
        wt_norm_tensor = wt_logits[residue_indices, seq_indices].unsqueeze(-1)
        LLR_tensor = wt_logits - wt_norm_tensor
        LLR_tensor_domain = LLR_tensor[:, aa_token_ids][dom_pos - 1:dom_pos - 1 + len(dom_seq), :]

        flattened_LLR = LLR_tensor_domain.flatten().to(device)
        flattened_exp = target_tensor.to(device)

        ft_tensor = torch.transpose(torch.stack([flattened_LLR, flattened_exp], dim=0), 0, 1)

        train_tensor = ft_tensor[train_indices]
        train_tensor = train_tensor[~torch.any(train_tensor.isnan(), dim=1)]
        positions = train_tensor[torch.randperm(len(train_tensor))[:num_samples]]

        loss = loss_fn(positions[:, 0], positions[:, 1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_tensor = ft_tensor[val_indices]
            val_tensor = val_tensor[~torch.any(val_tensor.isnan(), dim=1)]
            positions = val_tensor[torch.randperm(len(val_tensor))[:num_samples]]
            val_loss = loss_fn(positions[:, 0], positions[:, 1])
            val_losses.append(val_loss.item())

        if print_info:
            print(f"Epoch {epoch+1} - Train: {loss.item():.4f} | Val: {val_loss.item():.4f}")

    return model, train_losses, val_losses


def process_domain_batch(domain_ids):
    torch.set_num_threads(4)
    model_name = "hugohrban/progen2-medium"
    model, tokenizer = initialize_progen2_noeval(model_name)

    client = storage.Client()
    bucket = client.bucket('domainome-data')
    dict_dn_fitness_filtered = pickle.loads(
        bucket.blob('dict_dn_fitness_filtered.pkl').download_as_bytes()
    )
    dict_uniprot = pickle.loads(
        bucket.blob('dict_domainome_uniprot_new.pkl').download_as_bytes()
    )

    lora_config = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["qkv_proj", "out_proj"],
        lora_dropout=0.1, bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    for domain_id in domain_ids:
        print(f"[PID {os.getpid()}] Starting {domain_id}")

        dict_entry = dict_dn_fitness_filtered[domain_id]
        dom_pos = int(dict_entry['domain_start'])
        dom_seq = dict_entry['dom_seq']
        protein_seq = dict_uniprot[dict_entry['uniprot_id']]['sequence']
        fitness_tensor = torch.tensor(dict_entry['fitness'])

        base_model, tokenizer = initialize_progen2_noeval(model_name)

        model, train_losses, val_losses = FineTune_ProGen2_LORA(
            'cpu', base_model, tokenizer, lora_config,
            protein_seq, dom_seq, dom_pos, fitness_tensor,
            listwise_ranking_loss, lrate=1e-3, num_epochs=4,
            k=0.8, num_samples=10, print_info=False
        )

        print(f"[{domain_id}] Train: {train_losses} | Val: {val_losses}")

        model.eval()
        dict_uniprot_LLRs = dict_uniprot.copy()

        with torch.no_grad():
            for key in dict_uniprot_LLRs.keys():
                seq = dict_uniprot_LLRs[key]['sequence']
                if len(seq) > 1024:
                    seq = seq[:1023]
                lp, rlp, llr = collect_log_prob_pg2(seq, model, tokenizer)
                dict_uniprot_LLRs[key]['LLR'] = llr

        compressed_data = gzip.compress(pickle.dumps(dict_uniprot_LLRs))
        bucket.blob(f"lora_dicts/{domain_id}_lora_LLRs.pkl.gz").upload_from_string(compressed_data)

        del model, base_model, dict_uniprot_LLRs
        gc.collect()
        print(f"[PID {os.getpid()}] Done with {domain_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--total_domains", type=int, default=12)
    args = parser.parse_args()

    client = storage.Client()
    bucket = client.bucket('domainome-data')
    dict_dn_fitness_filtered = pickle.loads(
        bucket.blob('dict_dn_fitness_filtered.pkl').download_as_bytes()
    )

    keys = list(dict_dn_fitness_filtered.keys())[:args.total_domains]
    batches = [keys[i:i + args.batch_size] for i in range(0, len(keys), args.batch_size)]

    print(f"Total domains : {len(keys)}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Num batches   : {len(batches)}")
    print(f"Num workers   : {args.num_workers}")

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=args.num_workers) as pool:
        pool.map(process_domain_batch, batches)

    print("All domains complete.")
