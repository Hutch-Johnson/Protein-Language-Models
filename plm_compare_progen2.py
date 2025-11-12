import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, EsmForMaskedLM
from tokenizers import Tokenizer
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_list = [x for x in amino_acids]


def initialize_progen2(model_name):
    # Define tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    # Evaluate model
    model.eval()

    return model, tokenizer

def collect_log_prob_pg2(sequence, model, tokenizer, device="cpu"):
    # Define indices for log-likelihood ratio matrix
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_token_ids = [tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids]

    prompt1 = "1"+sequence
    prompt2 = "1"+sequence[::-1] 

    input_ids1 = torch.tensor(tokenizer.encode(prompt1)).unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits1 = model(input_ids1).logits

    input_ids2 =  torch.tensor(tokenizer.encode(prompt2)).unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits2 = model(input_ids2).logits

    logits2 = logits2[:, torch.arange(logits2.size(1) - 1, -1, -1), :]

    # take averages of matrices, 2nd one in reverse order
    # to simulate BERT output

    logits = (logits1 + logits2)/2
    
    log_probs = F.log_softmax(logits, dim = -1)
    n = log_probs.size(1)

    ref_log_probs = log_probs[0, torch.arange(input_ids1.size(1)), input_ids1[0]]
    ref_log_probs = ref_log_probs.unsqueeze(1)
    ref_log_probs = ref_log_probs[:n-1]

    log_probs = log_probs[0,:n-1]

    llr_matrix = log_probs - ref_log_probs
    llr_matrix = llr_matrix[:, aa_token_ids]
    log_probs = log_probs[:, aa_token_ids]

    return log_probs, ref_log_probs, llr_matrix


def seq_matrix_dict_pg2(sequence_list, model, tokenizer,device="cpu"):

    seq_dict = dict()

    n = len(sequence_list)

    for i in range(n):
        sequence = sequence_list[i]
        lp, rlp, llr = collect_log_prob_pg2(sequence,model,tokenizer)

        seq_dict[i] = {'sequence': sequence, 'log_probs': lp, 'ref_log_probs': rlp, 'llr_matrix': llr}

    return seq_dict


