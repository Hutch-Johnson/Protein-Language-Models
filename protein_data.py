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

from scipy.spatial.distance import jensenshannon

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


def prob_like_js(prob_dist1, prob_dist2, threshold=0.5):
    if jensenshannon(prob_dist1, prob_dist2) < threshold:
        return True
    return False


def plot_dist(prob_list, plot_range=(0,50)):
    eB = prob_list

    Blist = [np.array(eB)[i] for i in range(plot_range[0],plot_range[1])]

    cols = (plot_range[1] - plot_range[0] + 9)//10
    fig, axes = plt.subplots(10, cols, figsize=(25, 25))
    axes = axes.flatten()
    bar_width = 0.4
    x = torch.arange(1, 21)

    for i, ax in enumerate(axes):
        dist2 = Blist[i]
        # ax.bar(x - bar_width/2, dist1, width=bar_width, label="Dist 1", color="skyblue")
        ax.bar(x, dist2, width=bar_width, color="blue")
        ax.set_title(f"Comparison {i+1}")
        ax.set_xlabel("Amino Acids")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(aa_list)  
        ax.tick_params(axis='x', rotation=0)

    # Add a single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
