# Protein Language Model Study

This repo includes a collection of notebooks and code for comparing ESM2 and ProGen2 protein language models (PLMs).  The focus in on fine-tuning these PLMs using low rank adaptation (LoRA).

There is a .yml file `PMLenv.yml` containing the necessary Python packages.  The file `plm_compare_esm.py` and `plm_compare_progen2.py` contain fucntions for initiating ESM and ProGen models respectively.  Along with functions for storing log-probabilities matrices, ref_log_probs, llr_matrix into a dictionary.  Then there are functions for saving the dictionaries as .pickle files.

## Repo Structure

.

├── ColabRuns/                  # Google Colab experiments

├── Data/                       # Input datasets and processed files

├── Experiments/                # Experiment outputs and results

│
├── Configs_ESM2.ipynb          # ESM2 configuration notebook

├── ESM2_PFAM_boxplots.ipynb    # PFAM-level visualization

├── Stats_ESM2_PG2.ipynb        # Statistical analyses

│

├── plm_compare_esm.py          # ESM2 evaluation pipeline

├── plm_compare_progen2.py      # ProGen2 evaluation pipeline

├── pro_gen2_lora.py            # ProGen2 LoRA fine-tuning

├── protein_data.py             # Dataset loading and preprocessing

│

├── full_matrix_pg2.png         # Example visualization

├── PLMenv.yml                  # Conda environment

└── README.md

## Project Goals

Fine-tune proteinl language models on deep mutational scanning (DMS) data to improve the rank correlation between log-loss ratio matrices outputted by the model and the experimental DMS data.

`</> Bash`
`conda env create -f PLMenv.yml`
`conda activate PLMenv`

`from plm_compare_esm import *`

`from plm_compare_progen2 import *`
