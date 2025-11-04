import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, EsmForMaskedLM
from tokenizers import Tokenizer
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%
def initialize_esm2(model_name):
    # Define tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).to(device)
    # Evaluate mode
    model.eval()
    return model, tokenizer

def collect_log_prob_esm2(sequence, model, tokenizer):
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


# %%
def llr_heatmap(llr_matrix, positions=None, figsize=(15, 10), 
                cmap='RdBu_r',sequence='sequence'):

    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    if positions is None:
        positions = np.arange(llr_matrix.shape[0])
    else:
        positions = list(positions)
    plt.figure(figsize=figsize)
    sns.heatmap(llr_matrix[positions,:].T,
                            xticklabels=positions,
                            yticklabels=list(amino_acids),
                            cmap=cmap,
                            center=0,
                            cbar_kws={'label': 'LLR'})
    plt.xlabel(f'Position')
    plt.ylabel('Amino Acid')
    plt.title(f'Log-Likelihood Ratio Matrix \n {sequence}')
    plt.tight_layout()

    return plt


# %%
def seq_matrix_dict_esm2(sequence_list, model, tokenizer):

    seq_dict = dict()

    n = len(sequence_list)

    for i in range(n):
        sequence = sequence_list[i]
        lp, rlp, llr = collect_log_prob_esm2(sequence, model, tokenizer)

        seq_dict[i] = {'sequence': sequence, 'log_probs': lp, 'ref_log_probs': rlp, 'llr_matrix': llr}

    return seq_dict

# %%
# model_name = "facebook/esm2_t33_650M_UR50D"
# crx_sequence = "MMAYMNPGPHYSVNALALSGPSVDLMHQAVPYPSAPRKQRRERTTFTRSQLEELEALFAKTQYPDVYAREEVALKINLPESRVQVWFKNRRAKCRQQRQQQKQQQQPPGGQAKARPAKRKAGTSPRPSTDVCPDPLGISDSYSPPLPGPSGSPTTAVATVSIWSPASESPLPEAQRAGLVASGPSLTSAPYAMTYAPASAFCSSPSAYGSPSSYFSGLDPYLSPMVPQLGGPALSPLSGPSVGPSLAQSPTSLSGQSYGAYSPVDSLEFKDPTGTWKFTYNPMDPLDYKDQSAWKFQIL"

# %%
# model, tokenizer = initialize_esm2(model_name)

# %%
# seq_matrix_dict([crx_sequence], model, tokenizer)

# %%
# crx_lp, crx_rlp, crx_llr = collect_log_prob_esm2(crx_sequence, model, tokenizer)

# %%
# llr_heatmap(crx_llr,positions=range(20,50))


