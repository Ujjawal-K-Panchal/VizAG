"""
Title: Project Configuration.

Date: May 27, 2024; 7:22 PM

Author: Ujjawal K. Panchal & Ajinkya Chaudhari & Isha S. Joglekar
"""
import os
from pathlib import Path
import torch
import transformers, peft

#Important paths.
secrets = "./secrets"
modelstore = "./modelstore"

flikr8k = Path(modelstore) / "vizag-flikr8k"

if not os.path.exists(modelstore):
    os.makedirs(modelstore)
if not os.path.exists(secrets):
    os.makedirs(secrets)

with open(secrets + "/hf_token", "rt") as tokenfile:
    hf_token = tokenfile.read()

#CLIP config.
clip_model_name = "microsoft/git-base"

#Salesforce CLIP config.
sforceclip_model_name = "Salesforce/blip-image-captioning-large"



#RAG config.
batchsize = 128
k = 2
device = "cuda:0"
db_device = "cuda:1"
##Embedder config.
###it is recommended to set layers from 20 to 24.
emb_model_name = "mixedbread-ai/mxbai-embed-2d-large-v1"
layer_index = 22  # 1d: layer
embedding_size = 768  # 2d: embedding size
emb_pooling_strategy = "cls"


#Generator config.
llm_name = "meta-llama/Meta-Llama-3-8B-Instruct"
qtype = "qlora"
max_seq_len = 800
do_sample=True
temperature=0.1
top_p=0.9
##QLoRA Quantization config.
nf4_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype = torch.float16,
)
lora_config = peft.LoraConfig(
        r = 8,
        lora_alpha = 16,
        target_modules = ["q_proj", "v_proj"],
        bias = "none",
        task_type = "CAUSAL_LM",
)