# import os
# import gzip
# import gc
# import pickle

# from google.cloud import storage
# from dotenv import load_dotenv

# import pandas as pd
# import numpy as np
# import torch

# from transformers import AutoModelForCausalLM
# from transformers import AutoTokenizer, EsmForMaskedLM
# from tokenizers import Tokenizer
# from peft import get_peft_model, LoraConfig, TaskType

# from plm_compare_progen2 import *
# from protein_data import *
# from pro_gen2_lora import *

import os
import gzip
import gc
import pickle

from google.cloud import storage

import torch
from peft import LoraConfig, TaskType

from plm_compare_progen2 import *
from protein_data import *
from pro_gen2_lora import *


# load device and model
device = 'cpu'
print(f"Using {device} device")
model_name = "hugohrban/progen2-medium"
base_model, tokenizer = initialize_progen2_noeval(model_name)

# load fitness data filtered for ProGen2 context window of 1024
filename = '/Users/johnhutchens/Desktop/Practicum/Data/Domainome/dict_dn_fitness_filtered.pkl'
with open(filename, "rb") as f:
    dict_dn_fitness_filtered = pickle.load(f)

# load domainome data
filename = '/Users/johnhutchens/Desktop/Practicum/Data/Domainome/dict_domainome_uniprot_new.pkl'
with open(filename, "rb") as f:
    dict_uniprot = pickle.load(f)

# loop that saves LLR outputs from lora models as pkl.gz files of dictionaries
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["qkv_proj", "out_proj"],
    lora_dropout=0.1,
    bias="none",
    # task_type=TaskType.FEATURE_EXTRACTION
    task_type=TaskType.CAUSAL_LM
)


#loop to create models
for i in range(0,12):
    domain_id = keys[i]
    print(domain_id)
    dict_entry = dict_dn_fitness_filtered[domain_id]
    dom_pos = int(dict_entry['domain_start'])
    dom_seq = dict_entry['dom_seq']

    uniprot_id = dict_entry['uniprot_id']
    protein_seq = dict_uniprot[uniprot_id]['sequence']

    fitness_list = dict_entry['fitness']
    fitness_tensor = torch.tensor(fitness_list)


    loss = listwise_ranking_loss

    eps = 4
    lr = 1e-3
    num_samples = 10

    model, train_losses, val_losses = FineTune_ProGen2_LORA(device, base_model, tokenizer, 
                                            lora_config, protein_seq, dom_seq, dom_pos,
                                            fitness_tensor, loss, lrate=lr, num_epochs=eps, 
                                            k=0.8, num_samples=num_samples, print_info=False)

    print(f"Training loss = {train_losses}")
    print(f"Validation loss = {val_losses}")

    # set fine-tuned model to eval
    model.eval()   

    # new dictionary to store fine-tuned LLR
    dict_uniprot_LLRs = dict_uniprot.copy()

    with torch.no_grad():
        for key in dict_uniprot_LLRs.keys():
            seq = dict_uniprot_LLRs[key]['sequence']
            # print(len(seq))
            if len(seq) > 1024:
                seq = seq[:1023]
            lp, rlp, llr = collect_log_prob_pg2(seq, model, tokenizer)
            dict_uniprot_LLRs[key]['LLR'] = llr

    # delete model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # store in gcp bucket
    load_dotenv()
    cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cred_path

    client = storage.Client()
    bucket = client.bucket('domainome-data')

    data = dict_uniprot_LLRs
    compressed_data = gzip.compress(pickle.dumps(data))

    blob_name = domain_id+"_lora_LLRs.pkl.gz"
    blob = bucket.blob("lora_dicts/"+blob_name)

    blob.upload_from_string(compressed_data)

