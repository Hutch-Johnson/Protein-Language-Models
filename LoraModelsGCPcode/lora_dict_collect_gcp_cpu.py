import os
import gzip
import gc
import pickle
import torch
import torch.nn.functional as F
import numpy as np

from multiprocessing import Pool, cpu_count
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.cloud import storage

# from plm_compare_progen2 import *
# from protein_data import *
# from pro_gen2_lora import *

def initialize_progen2_noeval(model_name):
    '''
    Initializes the ProGen2 model with the given name.
    '''

    # Define tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # work around for google cloud vm
    if not hasattr(model, "all_tied_weights_keys"):
        model.all_tied_weights_keys = set()
    if not hasattr(model, "_tied_weights_keys"):
        model._tied_weights_keys = set()

    return model, tokenizer

def collect_log_prob_pg2(sequence, model, tokenizer, device="cpu"):
    '''
    Creates a log probability matrix for each position in the protein
    for the protein with given sequence, using the given ProGen2 model
    and tokenizer.  Device is by default cpu but can be changed if using
    GPU or other device.  Outputs log probability matrix, reference log 
    probability matrix and log loss ratio matrix.
    '''
    # Define indices for log-likelihood ratio matrix
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_token_ids = [tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids]

    prompt1 = "1"+sequence  # run it forwards
    prompt2 = "2"+sequence[::-1]  # run it backwards

    input_ids1 = torch.tensor(tokenizer.encode(prompt1)).unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits1 = model(input_ids1).logits
    shift_logits1 = logits1[:, :-1, :]  # remove last entry

    input_ids2 =  torch.tensor(tokenizer.encode(prompt2)).unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits2 = model(input_ids2).logits
    shift_logits2 = logits2[:, :-1, :] # remove last entry

    shift_logits2 = shift_logits2[:, torch.arange(shift_logits2.size(1) - 1, -1, -1), :]

    input_ids = input_ids1[:, 1:]

    # take averages of matrices, 2nd one in reverse order
    # to simulate BERT output

    # check on this???

    logits = (shift_logits1 + shift_logits2)/2

    log_probs = F.log_softmax(logits, dim = -1)
    # n = log_probs.size(1)

    ref_log_probs = log_probs[0, torch.arange(input_ids.size(1)), input_ids[0]]
    ref_log_probs = ref_log_probs.unsqueeze(1)
    #ref_log_probs = ref_log_probs[:n-1]

    #log_probs = log_probs[0,:n-1]

    llr_matrix = log_probs - ref_log_probs
    llr_matrix = llr_matrix[0][:, aa_token_ids]
    log_probs = log_probs[0][:, aa_token_ids]

    return np.array(log_probs), np.array(ref_log_probs), np.array(llr_matrix)

def insert_wt(seq, pos, wt_aa):
    seq_list = list(seq)
    pos = int(pos)
    if pos < len(seq_list):
        seq_list[pos] = wt_aa
    return ''.join(seq_list)

# computes the ranking loss between two iterables
def listwise_ranking_loss(preds, targets):
    indices = targets.sort(descending=True).indices
    preds = torch.gather(preds, dim=-1, index=indices)
    cumsums = preds.exp().flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
    loss = torch.log(cumsums + 1e-10) - preds
    return loss.mean()

def FineTune_ProGen2_LORA(device, base_model, tokenizer, lora_config, 
                          protein_seq, dom_seq, dom_pos, target_tensor, loss_fn, 
                          lrate=-1e-5, num_epochs=5, k=0.8, 
                          num_samples=20, print_info=True):
    '''
    device is GPU or CPU
    base_model is ProGen2 model
    tokenizer is ProGen2 tokenizer
    lora_config is data for peft lora fine tuning 
    lr is learning rate for gradient descent on new layer
    protein sequence is the sequence lora layer is being trained on
    target_tensor is what loss is measured against initially this is the actual protein sequence
    loss_fn is loss function used in training and validation
    num_epochs is the number of training steps
    k is the proportion of sequence used in training. validation and test indices are created as half the remaining indices each
    num_samples is the number of sample drawn for each epoch used for training and validation
    '''
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_token_ids = [tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids]

    model = get_peft_model(base_model, lora_config)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lrate)

    exp_tensor = target_tensor
    seq_len = exp_tensor.shape[0]

    all_indices = torch.randperm(seq_len)

    num_train = int(seq_len * k)

    train_indices = all_indices[:num_train] 
    val_indices = all_indices[num_train:]

    train_losses = []
    val_losses = []
    early_stop_count = 0


    vocab_dict = tokenizer.get_vocab()
    seq_list = list(protein_seq)
    inputs = tokenizer(protein_seq, return_tensors="pt").to(device)

    for epoch in range(num_epochs):

        model.train()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)  # (seq_len, vocab_size)
        wt_logits = torch.log_softmax(logits, dim=-1)
        residue_indices = torch.arange(len(protein_seq))

        # ignoring BOS token
        seq_indices = [vocab_dict[aa] for aa in seq_list]
        wt_norm_tensor = wt_logits[residue_indices, seq_indices].unsqueeze(-1)
        LLR_tensor = wt_logits - wt_norm_tensor
        LLR_tensor_aa_only = LLR_tensor[:, aa_token_ids]
        LLR_tensor_domain = LLR_tensor_aa_only[dom_pos-1:dom_pos-1+len(dom_seq),:]

        # flatten the LLR_tensor
        # flattened_LLR_tensor = LLR_tensor_aa_only.flatten()

        flattened_LLR_tensor = LLR_tensor_domain.flatten()
        flattened_exp_tensor = exp_tensor.to(device)
        flattened_LLR_tensor = flattened_LLR_tensor.to(device)

        #to stack
        combined = torch.stack([flattened_LLR_tensor, flattened_exp_tensor], dim=0)
        ft_tensor = torch.transpose(combined, 0, 1)

        # predicted_scores = []
        # experimental_values = []

        train_tensor = ft_tensor[train_indices]

        #drop nan
        train_tensor = train_tensor[~torch.any(train_tensor.isnan(), dim=1)]

        # num_samples = num_samples
        positions = train_tensor[torch.randperm(len(train_tensor))[:num_samples]]

        predicts = positions[:, 0] #LLR
        targets = positions[:, 1] #exp

        # Compute loss with predicts and targets
        loss = loss_fn(predicts, targets)
        # log_probs = torch.log_softmax(logits, dim=-1)
        # loss = -log_probs[torch.arange(len(seq_indices)), seq_indices].mean() # needs to be difference of predict/target
        loss.backward()
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.abs().mean())
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.item())

        # ----------- VALIDATION (no backprop) -----------
        model.eval()
        with torch.no_grad():
            val_tensor = ft_tensor[val_indices]
            val_tensor = val_tensor [~torch.any(val_tensor.isnan(), dim=1)]

            # num_samples = 45
            positions = val_tensor[torch.randperm(len(val_tensor))[:num_samples]]

            predicts = positions[:, 0] #LLR
            targets = positions[:, 1] #exp

            val_loss = loss_fn(predicts, targets)
            # val_loss = listwise_ranking_loss(predicts, targets)
            #   val_log_probs = torch.log_softmax(logits, dim=-1)
            #   val_loss = -log_probs[torch.arange(len(seq_indices)), seq_indices].mean()
            val_losses.append(val_loss.item())

        if print_info==True:
            print(f"Epoch {epoch+1} - Training Loss: {loss.item():.4f} | Validation Loss: {val_loss.item():.4f}")

            if val_loss.item() > loss.item():
                early_stop_count += 1
            else:
                early_stop_count = 0
                
            print(f"Validation loss has exceeded training loss {early_stop_count} time(s) in a row")
            # if early_stop_count > 2:
            #     print("Validation loss exceeded training loss 3 times — early stopping.")
            #     break

    return model, train_losses, val_losses
# , ft_tensor, test_indices



import os
import gzip
import gc
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from multiprocessing import Pool, cpu_count
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.cloud import storage

# ---- All your existing functions go here (unchanged) ----
# initialize_progen2_noeval, collect_log_prob_pg2, listwise_ranking_loss, FineTune_ProGen2_LORA

def process_domain(domain_id):
    """Worker function — each process handles one domain independently."""
    print(f"[PID {os.getpid()}] Starting {domain_id}")

    # Each worker loads its own model copy (no sharing across processes)
    model_name = "hugohrban/progen2-medium"
    base_model, tokenizer = initialize_progen2_noeval(model_name)

    # Each worker gets its own GCS client
    client = storage.Client()
    bucket = client.bucket('domainome-data')

    blob = bucket.blob('dict_dn_fitness_filtered.pkl')
    dict_dn_fitness_filtered = pickle.loads(blob.download_as_bytes())

    blob = bucket.blob('dict_domainome_uniprot_new.pkl')
    dict_uniprot = pickle.loads(blob.download_as_bytes())

    dict_entry = dict_dn_fitness_filtered[domain_id]
    dom_pos = int(dict_entry['domain_start'])
    dom_seq = dict_entry['dom_seq']
    uniprot_id = dict_entry['uniprot_id']
    protein_seq = dict_uniprot[uniprot_id]['sequence']
    fitness_tensor = torch.tensor(dict_entry['fitness'])

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["qkv_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

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
    blob = bucket.blob(f"lora_dicts/{domain_id}_lora_LLRs.pkl.gz")
    blob.upload_from_string(compressed_data)

    del model, base_model, dict_uniprot_LLRs
    gc.collect()
    print(f"[PID {os.getpid()}] Done with {domain_id}")


if __name__ == "__main__":
    client = storage.Client()
    bucket = client.bucket('domainome-data')
    dict_dn_fitness_filtered = pickle.loads(
        bucket.blob('dict_dn_fitness_filtered.pkl').download_as_bytes()
    )
    keys = list(dict_dn_fitness_filtered.keys())[:12]

    # Use however many workers you want — start conservative
    num_workers = min(6, cpu_count())
    print(f"Launching pool with {num_workers} workers for {len(keys)} domains")

    with Pool(processes=num_workers) as pool:
        pool.map(process_domain, keys)

    print("All domains complete.")

