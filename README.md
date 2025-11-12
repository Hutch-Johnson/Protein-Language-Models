## About this repo

This repo includes a collection of notebooks containing code for comparing different protein language models.

There is a .yml file `PMLenv.yml` containing the necessary Python packages.  The file `plm_compare_esm.py` and `plm_compare_progen2.py` contain fucntions for initiating ESM and ProGen models respectively.  Along with functions for storing log-probabilities matrices, ref_log_probs, llr_matrix into a dictionary.  Then there are functions for saving the dictionaries as .pickle files.

You can import each by running 

`from plm_compare_esm import *`

`from plm_compare_progen2 import *`
