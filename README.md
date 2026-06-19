# Protein Language Model Study

This repo includes a collection of notebooks and code for comparing ESM2 and ProGen2 protein language models (PLMs).  The focus in on fine-tuning these PLMs using low rank adaptation (LoRA).

There is a .yml file `PMLenv.yml` containing the necessary Python packages.  The file `plm_compare_esm.py` and `plm_compare_progen2.py` contain fucntions for initiating ESM and ProGen models respectively.  Along with functions for storing log-probabilities matrices, ref_log_probs, llr_matrix into a dictionary.  Then there are functions for saving the dictionaries as .pickle files.


## Repository Structure

```text
.
├── ColabRuns/                      # Google Colab notebooks and exploratory analyses
├── Data/                           # Raw and processed protein datasets
├── Experiments/                    # Experiment outputs, figures, and model checkpoints
│
├── Configs_ESM2.ipynb              # ESM2 configuration and setup notebook
├── ESM2_PFAM_boxplots.ipynb        # PFAM-level visualization and analysis
├── Stats_ESM2_PG2.ipynb            # Statistical comparisons between ESM2 and ProGen2
│
├── plm_compare_esm.py              # ESM2 evaluation pipeline
├── plm_compare_progen2.py          # ProGen2 evaluation pipeline
├── pro_gen2_lora.py                # ProGen2 LoRA fine-tuning script
├── protein_data.py                 # Data loading and preprocessing utilities
│
├── full_matrix_pg2.png             # Example ProGen2 results visualization
├── PLMenv.yml                      # Conda environment specification
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation
```

## Project Goals

Fine-tune proteinl language models on deep mutational scanning (DMS) data to improve the rank correlation between log-loss ratio matrices outputted by the model and the experimental DMS data.

```</> Bash
conda env create -f PLMenv.yml
conda activate PLMenv```

```bash
from plm_compare_esm import *
from plm_compare_progen2 import *
```
