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
    
    ref_log_probs = log_probs[0, torch.arange(input_ids.size(1)), input_ids[0]]
    ref_log_probs = ref_log_probs.unsqueeze(1)

    llr_matrix = log_probs - ref_log_probs
    llr_matrix = llr_matrix[0][:, aa_token_ids]
    log_probs = log_probs[0][:, aa_token_ids]

    return np.array(log_probs), np.array(ref_log_probs), np.array(llr_matrix)


##############################################################################
##############################################################################
### This version uses conditional log probabilities and no batches ###########
##############################################################################
##############################################################################

# old version something is off

# def collect_logprob_pg2_cond(sequence, model, tokenizer):
#     """
#     Creates a log probability matrix conditionally based on the previous tokens.  
#     The log probability matrix is used to make a reference log probability matrix.
#     The log-loss ratio matrix is constructed from the log probability matrix and
#     reference log probability matrix.
#     """
#     amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
#     aa_to_id = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids}
#     aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    
#     log_probs_matrix = []

#     for i in range(len(sequence)):
        
#         if i == 0:
#             prompt = tokenizer.bos_token  # or "1
#         else:
#             prompt = tokenizer.bos_token + sequence[:i]
        
#         # prompt = "1"+sequence
        
#         # Encode prompt
#         input_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False)).unsqueeze(0).to(model.device)
        
#         # Forward pass
#         with torch.no_grad():
#             logits = model(input_ids).logits  
        
#         # Logits for next position
#         next_token_logits = logits[0, -1, :]  
#         log_probs = F.log_softmax(next_token_logits, dim=-1)
        
#         # Extract log-probs for all 20 amino acids
#         position_log_probs = [log_probs[aa_to_id[aa]].item() for aa in amino_acids]
#         log_probs_matrix.append(position_log_probs)
    
#     log_probs_matrix = torch.tensor(log_probs_matrix, device=model.device)
    
#     true_ids = torch.tensor([aa_to_index[aa] for aa in sequence], device=model.device)
    
#     ref_log_probs = log_probs_matrix[torch.arange(len(sequence)), true_ids].unsqueeze(1)
#     llr_matrix = log_probs_matrix - ref_log_probs
    
#     return log_probs_matrix.cpu().numpy(), ref_log_probs.cpu().numpy(), llr_matrix.cpu().numpy()

def collect_logprob_pg2_cond(sequence, model, tokenizer, batch_size=None):
    """
    Creates a log probability matrix conditionally based on previous tokens.
    
    Returns:
        log_probs_matrix: (seq_len, 20) log probabilities for each AA at each position
        ref_log_probs: (seq_len, 1) log probability of true AA at each position
        llr_matrix: (seq_len, 20) difference from reference
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_id = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids}
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    
    log_probs_list = []
    
    for i in range(len(sequence)):
        # prompt = tokenizer.bos_token + (sequence[:i] if i > 0 else "")
        prompt = "1"+(sequence[:i] if i > 0 else "")
        
        input_ids = torch.tensor(
            tokenizer.encode(prompt, add_special_tokens=False)
        ).unsqueeze(0).to(model.device)
        
        with torch.no_grad():
            logits = model(input_ids).logits
            next_token_logits = logits[0, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            
            position_log_probs = torch.tensor(
                [log_probs[aa_to_id[aa]].item() for aa in amino_acids],
                device=model.device
            )
            log_probs_list.append(position_log_probs)
    
    log_probs_matrix = torch.stack(log_probs_list)
    true_ids = torch.tensor([aa_to_index[aa] for aa in sequence], device=model.device)
    ref_log_probs = log_probs_matrix[torch.arange(len(sequence)), true_ids].unsqueeze(1)
    llr_matrix = log_probs_matrix - ref_log_probs
    
    return (
        log_probs_matrix.cpu().numpy(),
        ref_log_probs.cpu().numpy(),
        llr_matrix.cpu().numpy()
    )


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


