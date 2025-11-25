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

def initialize_esm2(model_name, device="cpu"):
    '''
    Initializes the ESM2 model with the given name.
    '''
    # Define tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).to(device)
    # Evaluate mode
    model.eval()
    return model, tokenizer

def collect_log_prob_esm2(sequence, model, tokenizer):
    '''
    Creates a log probability matrix for each position in the protein
    for the protein with given sequence, using the given ESM2 model
    and tokenizer.  Device is by default cpu but can be changed if using
    GPU or other device.  Outputs log probability matrix, reference log 
    probability matirx and log loss ratio matrix.
    '''
    # Define indices for log-likelihood ratio matrix
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_token_ids = [tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids]

    # Clean sequence
    sequence.replace('\n','')
    sequence.replace(' ','')

    # Tokenize sequence for model
    inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)

    # Output predcitions from model, do not compute gradients
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    input_ids = inputs['input_ids']

    # Define tensor of log probabilities for each
    # amino acid in each position of the sequence
    log_probs = torch.log_softmax(logits, dim=-1)

    # Define tensor of reference log probabilities
    # the log probabilities for actual amino acid
    # in sequence
    ref_log_probs = log_probs[0, torch.arange(input_ids.size(1)), input_ids[0]]
    # Resize ref_log_probs
    ref_log_probs = ref_log_probs.unsqueeze(1)
    log_probs = log_probs[0]

    # Define the log-likelihood ratio matrix
    llr_matrix = log_probs - ref_log_probs
    llr_matrix = llr_matrix[1:-1,:]
    llr_matrix = llr_matrix[:, aa_token_ids]
    log_probs = log_probs[1:-1,:]
    log_probs = log_probs[:, aa_token_ids]
    ref_log_probs = ref_log_probs[1:-1]

    return log_probs, ref_log_probs, llr_matrix


def seq_matrix_dict_esm2(sequence_list, model, tokenizer):
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
        lp, rlp, llr = collect_log_prob_esm2(sequence, model, tokenizer)

        seq_dict[i] = {'sequence': sequence, 'log_probs': lp, 'ref_log_probs': rlp, 'llr_matrix': llr}

    return seq_dict
