import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training
from dataclasses import field
import random
import pdb
from tqdm import tqdm
from datasets import Dataset
import re
# os.environ["WANDB_DISABLED"] = "true"

output_dir="xxxxxx"
cache_dir="xxxxxx"

os.makedirs(output_dir,exist_ok=True)
os.makedirs(cache_dir,exist_ok=True)

model_name = "meta-llama/Meta-Llama-3-8B"


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--language", type=str, default="English")
parser.add_argument("--task", type=str, default="Wiki")

args = parser.parse_args()
print(args)

import ast

def retrive_neuron(filename):
    # Empty list to store the dictionaries
    activate_neuron = []

    # Open the file and read line by line
    with open(filename, 'r') as file:
        neurons = file.readlines()
        for neuron in neurons:
            neuron = eval(neuron.strip())
            activate_neuron.append(neuron)

    return activate_neuron

def deduplicate(neuron_target, neuron_delete):
    index_keys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    for key in index_keys:
        neuron_target[0][key] = neuron_target[0][key] - neuron_delete[0][key]
        neuron_target[1][key] = neuron_target[1][key] - neuron_delete[1][key]
        neuron_target[2][key] = neuron_target[2][key] - neuron_delete[2][key]
        neuron_target[3][key] = neuron_target[3][key] - neuron_delete[3][key]
        neuron_target[4][key] = neuron_target[4][key] - neuron_delete[4][key]

    return neuron_target


activate_neuron =  retrive_neuron('xxxxxx')


dataset = load_dataset("json", data_files="xxxxxxx",split="train", cache_dir=cache_dir)


# base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")


base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

for name, param in tqdm(base_model.named_parameters()):
    param.requires_grad = True


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# data['original_question']+data['response']

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['original_question'])):
        text = f"{example['original_question'][i]}. {example['response'][i]}"
        # text = f"Question: {example['prompt'][i]}\nAnswer: {example['completion'][i]}"
        # text = f"```{example['prompt'][i]}{example['completion'][i]}```"
        output_texts.append(text)
    return output_texts


# Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    gradient_checkpointing =True,
    max_grad_norm= 0.3,
    num_train_epochs=1, 
    learning_rate=2e-6,
    bf16=True,
    save_steps=500,
    save_total_limit=0,
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    activate_neuron=activate_neuron,
)

trainer = SFTTrainer(
    base_model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=512,
    formatting_func=formatting_prompts_func,
    args=training_args
)

trainer.train() 

output_dir = os.path.join(output_dir, "Llama3_Reason")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)