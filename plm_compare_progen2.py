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

##############################################################################
##############################################################################
############### ProGen2 forward or backward pass only ########################
##############################################################################

def collect_log_prob_pg2_forward(sequence, model, tokenizer, device="cpu"):
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
    #prompt2 = "2"+sequence[::-1]  # run it backwards

    input_ids1 = torch.tensor(tokenizer.encode(prompt1)).unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits1 = model(input_ids1).logits
    shift_logits1 = logits1[:, :-1, :]  # remove last entry

    # input_ids2 =  torch.tensor(tokenizer.encode(prompt2)).unsqueeze(0).to(model.device)
    # with torch.no_grad():
    #     logits2 = model(input_ids2).logits
    # shift_logits2 = logits2[:, :-1, :] # remove last entry

    # shift_logits2 = shift_logits2[:, torch.arange(shift_logits2.size(1) - 1, -1, -1), :]

    input_ids = input_ids1[:, 1:]

    # take averages of matrices, 2nd one in reverse order
    # to simulate BERT output

    # logits = (shift_logits1 + shift_logits2)/2
    logits = shift_logits1

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

##############################################################################

def collect_log_prob_pg2_backward(sequence, model, tokenizer, device="cpu"):
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
    # with torch.no_grad():
    #     logits1 = model(input_ids1).logits
    # shift_logits1 = logits1[:, :-1, :]  # remove last entry

    input_ids2 =  torch.tensor(tokenizer.encode(prompt2)).unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits2 = model(input_ids2).logits
    shift_logits2 = logits2[:, :-1, :] # remove last entry

    shift_logits2 = shift_logits2[:, torch.arange(shift_logits2.size(1) - 1, -1, -1), :]

    input_ids = input_ids1[:, 1:]

    # logits of backward pass

    logits = shift_logits2

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


##############################################################################
##############################################################################
### This version uses conditional log probabilities and no batches


def collect_logprob_pg2_cond(sequence, model, tokenizer):
    """
    Get log probability matrix for a protein sequence.
    Computes each position based conditionally on all previous positions.
    
    Returns:
        log_probs_matrix: (seq_len, 20) log-probs for all amino acids
        ref_log_probs: (seq_len, 1) log-probs of the true amino acids
        llr_matrix: (seq_len, 20) log-likelihood ratios relative to true AA
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Token IDs for model lookup
    aa_to_id = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids}
    # Column index in our log_probs_matrix
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    
    log_probs_matrix = []

    for i in range(len(sequence)):
        # Build prompt: BOS/start token + sequence prefix
        if i == 0:
            prompt = tokenizer.bos_token  # or "1" if your tokenizer expects that
        else:
            prompt = tokenizer.bos_token + sequence[:i]
        
        # Encode prompt
        input_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False)).unsqueeze(0).to(model.device)
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_ids).logits  # shape: (1, seq_len, vocab_size)
        
        # Logits for next position
        next_token_logits = logits[0, -1, :]  # shape: (vocab_size,)
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        
        # Extract log-probs for all 20 amino acids
        position_log_probs = [log_probs[aa_to_id[aa]].item() for aa in amino_acids]
        log_probs_matrix.append(position_log_probs)
    
    log_probs_matrix = torch.tensor(log_probs_matrix, device=model.device)
    
    # True amino acid indices for columns
    true_ids = torch.tensor([aa_to_idx[aa] for aa in sequence], device=model.device)
    
    # Reference log-probs of the sequence
    ref_log_probs = log_probs_matrix[torch.arange(len(sequence)), true_ids].unsqueeze(1)
    
    # Log-likelihood ratio
    llr_matrix = log_probs_matrix - ref_log_probs
    
    return log_probs_matrix.cpu().numpy(), ref_log_probs.cpu().numpy(), llr_matrix.cpu().numpy()

##### This version uses conditional log probabilities and batches



##############################################################################
##############################################################################


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


