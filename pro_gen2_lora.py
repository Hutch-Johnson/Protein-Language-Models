import os

from io import StringIO
from google.cloud import storage
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, EsmForMaskedLM
from tokenizers import Tokenizer
from peft import get_peft_model, LoraConfig, TaskType


# some functions
# inserts mutation amino acid in corresponding position in protein sequence
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
    lr is learning rate for gradien descent on new layer
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
    # num_val = seq_len - num_train

    # num_val   = int(seq_len * k/2)
    # num_test  = seq_len - num_train - num_val

    train_indices = all_indices[:num_train] 
    val_indices = all_indices[num_train:]
    # val_indices   = all_indices[num_train:num_train + num_val]
    # test_indices  = all_indices[num_train + num_val:]

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