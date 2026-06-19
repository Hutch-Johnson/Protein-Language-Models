import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, EsmForMaskedLM
from tokenizers import Tokenizer
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


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
    '''
    Determines the Jensen-Shannon distance bewteen 
    prob_dist1 and prob_dist2 and determines if it
    is smaller than threshold.  Produces boolean
    based on outcome.
    '''
    if jensenshannon(prob_dist1, prob_dist2) < threshold:
        return True
    return False


def js_non_prob_list(prob_dist1_list, prob_dist2=np.ones(20)/20, threshold=0.5):
    output=[]
    n = len(prob_dist1_list)
    for i in range(n):
        prob_dist1 = prob_dist1_list[i]
        if not prob_like_js(prob_dist1,prob_dist2,threshold):
            output.append(i)
    return(sorted(output))


def plot_dist(prob_list, plot_range=(0,50)):
    '''
    Takes a list of probabilities, prob_list and plots
    a bar chart of their distribution for the given 
    range.
    '''
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


def wild_type(mut_name, sequence):
    '''
    Takes protein mutation name of the form
    [A12G:D4E:...] and creates the original 
    wild type sequence.
    '''

    mut_list = mut_name.split(":")

    for x in mut_list:
        mut = x
        len_mut = len(mut)
        orig = mut[0]
        pos = int(mut[1:len_mut-1])-1
        new = mut[len_mut-1]

        if new==sequence[pos]:
            # print(f"Position {pos + 1} changed from {new} to {orig}.")
            wild_seq = sequence[:pos] + orig + sequence[pos+1:]
        else:
            return f"Amino acid {new} not in position {pos + 1}."
        
    return wild_seq


def spearman_ignore_nan(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    return spearmanr(a[mask], b[mask])


def make_mutation_fitness_df(domain_id, df):
    """
    Docstring for make_mutation_fitness_df
    
    :param domain_id: domain id from domainome
    :param df: dataframe of domainome data

    Returns a dataframe with wild type sequence for the domain
    sequence and fitness data from DMS experiments as a 
    column
    """

    domain_id_list=domain_id.split("_")
    dom_pos = float(domain_id_list[-1])

    df_one_protein = df.where(df['domain_ID'] == domain_id).dropna()

    dom_position = df_one_protein['position'] - dom_pos
    df_one_protein.insert(loc=0, column='real_position', value=dom_position)
    df_one_protein_ns = df_one_protein[df_one_protein['mut_aa'] != '*'].copy()

    df_one_protein_ns['wt_seq'] = df_one_protein_ns.apply(lambda row: 
                                                      insert_wt(row['aa_seq'], row['real_position'], row['wt_aa']),
                                                      axis=1)
    
    df_mutation = df_one_protein_ns[['wt_seq','real_position','mut_aa','normalized_fitness']]

    return df_mutation

