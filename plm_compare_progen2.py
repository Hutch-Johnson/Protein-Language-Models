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
    '''
    Initializes the ProGen2 model with the given name.
    '''
    # Define tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    # Evaluate model
    model.eval()

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


def seq_matrix_dict_pg2(sequence_list, model, tokenizer,device="cpu"):
    '''
    Takes a list of protein sequences and using the given model, tokenizer,
    and device makes a dictionary of the sequence, log probability matrix, 
    reference log probability matrix, and log loss ratio matrix using given
    ProGen2 model.
    '''

    seq_dict = dict()

    n = len(sequence_list)

    for i in range(n):
        sequence = sequence_list[i]
        lp, rlp, llr = collect_log_prob_pg2(sequence,model,tokenizer)

        seq_dict[i] = {'sequence': sequence, 'log_probs': lp, 'ref_log_probs': rlp, 'llr_matrix': llr}

    return seq_dict


