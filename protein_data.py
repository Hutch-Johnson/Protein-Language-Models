import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, EsmForMaskedLM
from tokenizers import Tokenizer
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

def llr_heatmap(llr_matrix, positions=None, figsize=(15, 10), 
                cmap='RdBu_r',sequence='sequence'):
    '''
    Produces a log likelihood ratio matrix heat map with the
    positions in the protein as the x-axis and the amino acids
    as the y-axis.
    Inputs: llr_matrix, a list of positions to display, figsize,
    cmap and the protein sequence
    Outputs: matplotlib plot heatmap
    '''

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

def pickle_plm_matrices(dict, filename):

    '''
    Saves a dictionary to a pickle file.
    '''

    with open(filename, 'wb') as f:
        pickle.dump(dict, f, protocol=pickle.HIGHEST_PROTOCOL)

