import os
from dataclasses import field, dataclass
from typing import Optional, Any
import sys

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

from rouge_score import rouge_scorer
from transformers import AutoTokenizer
import random
from itertools import groupby
import pdb
import re
import multiprocessing
import threading
import json
from concurrent.futures import ThreadPoolExecutor



from tqdm import tqdm


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from peft import PeftModel
from typing import List
import logging
logging.basicConfig(level=logging.INFO)

random.seed(112)

import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")


# base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"

def deduplicate(neuron_target, neuron_delete):
    index_keys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]           
    for key in index_keys:
        neuron_target[0][key] = neuron_target[0][key] - neuron_delete[0][key]
        neuron_target[1][key] = neuron_target[1][key] - neuron_delete[1][key]
        neuron_target[2][key] = neuron_target[2][key] - neuron_delete[2][key]
        neuron_target[3][key] = neuron_target[3][key] - neuron_delete[3][key]
        neuron_target[4][key] = neuron_target[4][key] - neuron_delete[4][key]

    return neuron_target

def intersection(neuron_target, neuron_delete):
    index_keys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]           
    for key in index_keys:
        neuron_target[0][key] = neuron_target[0][key] & neuron_delete[0][key]
        neuron_target[1][key] = neuron_target[1][key] & neuron_delete[1][key]
        neuron_target[2][key] = neuron_target[2][key] & neuron_delete[2][key]
        neuron_target[3][key] = neuron_target[3][key] & neuron_delete[3][key]
        neuron_target[4][key] = neuron_target[4][key] & neuron_delete[4][key]

    return neuron_target


def Prompting(instruction, question,activate_keys_fwd_up_set,
                activate_keys_fwd_down_set,
                activate_keys_q_set,
                activate_keys_k_set,
                activate_keys_v_set, 
                under_layer, 
                gen_layer, 
                atten_number,
                ffn_number,
                whether_under,
                whether_reason,
                whether_gen,
                whether_under_fwd,
                whether_reason_fwd,
                whether_gen_fwd):

    prompt = question + instruction

    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**{'input_ids':inputs.input_ids, 'max_new_tokens':512},
                activate_keys_fwd_up_set=activate_keys_fwd_up_set,
                activate_keys_fwd_down_set=activate_keys_fwd_down_set,
                activate_keys_q_set=activate_keys_q_set,
                activate_keys_k_set=activate_keys_k_set,
                activate_keys_v_set=activate_keys_v_set, 
                under_layer=under_layer, 
                gen_layer=gen_layer, 
                atten_number=atten_number,
                ffn_number=ffn_number,
                whether_under=whether_under,
                whether_reason=whether_reason,
                whether_gen=whether_gen,
                whether_under_fwd=whether_under_fwd,
                whether_reason_fwd=whether_reason_fwd,
                whether_gen_fwd=whether_gen_fwd)
    answer = tokenizer.decode(outputs[0]).replace('</s>', '').replace(prompt, '')

    print(answer)

    try:
        answer = re.findall(r'####\s(.+)', answer)[0]
        prd = re.findall(r"\d+\,?\.?\d*",answer)[-1]
        prd = float(prd.replace(',', '').rstrip('.')) if prd else prd
        prd = int(prd)
        answer = int(prd)

    except:
        try:
            prd = re.findall(r"\d+\,?\.?\d*",answer)[-1]
            prd = float(prd.replace(',', '').rstrip('.')) if prd else prd
            answer = int(prd)
        except:
            answer = -1
    
    return answer


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

def main(argv):
    dataset_en = list(load_dataset("juletxara/mgsm", 'en')["test"])

    dataset_small = list(load_dataset("juletxara/mgsm", argv[0])["test"])


    correct_en = 0
    correct_small = 0
    all_index = 0
    
    full_name = {'zh':'chinese', 'fr':'french', 'de':'german', 'ru':'russian', 'es':'spanish', 'en':'english'}
    task_instruction_set = {'zh':"请逐步解答。\n", 'de':"Lassen Sie uns dies Schritt für Schritt durchgehen.\n", 'fr':"Voyons cela étape par étape.\n", 'es':"Vayamos paso a paso.\n", 'ru':"Давайте это шаг за шагом.\n"}
    prompt_set_1 = {'zh':"问题：", 'de':"Frage: ", 'fr':"Question: ", 'es':"Pregunta: ", 'ru':"вопрос: "}
    prompt_set_2 = {'zh':'\n答案：', 'de':'\nAntwort: ', 'fr':'\nRépondre: ', 'es':'\nRespuesta: ', 'ru':'\nотвечать: '}


    neuron_path = "./output_mixtral/"+full_name[argv[0]] + "_all.txt"

    activate_neuron =  retrive_neuron(neuron_path)

    english_neuron =  retrive_neuron("./output_mixtral/english_all.txt")
    spanish_neuron =  retrive_neuron("./output_mixtral/spanish_all.txt")
    french_neuron =  retrive_neuron("./output_mixtral/french_all.txt")
    chinese_neuron =  retrive_neuron("./output_mixtral/chinese_all.txt")
    russian_neuron =  retrive_neuron("./output_mixtral/russian_all.txt")

    english_neuron = intersection(english_neuron, spanish_neuron)
    english_neuron = intersection(english_neuron, french_neuron)
    english_neuron = intersection(english_neuron, chinese_neuron)
    english_neuron = intersection(english_neuron, russian_neuron)

    activate_neuron  = deduplicate(activate_neuron , english_neuron)


    activate_keys_fwd_up_set = activate_neuron[0]
    activate_keys_fwd_down_set = activate_neuron[1]
    activate_keys_q_set = activate_neuron[2]
    activate_keys_k_set = activate_neuron[3]
    activate_keys_v_set = activate_neuron[4]


    with tqdm(total=len(dataset_en)) as pbar:
        for i in range(len(dataset_en)):
            all_index += 1
            data_en = dataset_en[i]
            data_small = dataset_small[i]

            task_instruction_en = """Let\'s this step by step.\n"""
            prompt_en = "Question: " + data_en['question'] + '\nAnswer: '
            answer_en = Prompting(task_instruction_en, prompt_en, 
                activate_keys_fwd_up_set=activate_keys_fwd_up_set,
                activate_keys_fwd_down_set=activate_keys_fwd_down_set,
                activate_keys_q_set=activate_keys_q_set,
                activate_keys_k_set=activate_keys_k_set,
                activate_keys_v_set=activate_keys_v_set, 
                under_layer=int(argv[1]), 
                gen_layer=int(argv[2]), 
                atten_number=int(argv[3]),
                ffn_number=int(argv[4]),
                whether_under=bool(argv[5]=='True'),
                whether_reason=bool(argv[6]=='True'),
                whether_gen=bool(argv[7]=='True'),
                whether_under_fwd=bool(argv[8]=='True'),
                whether_reason_fwd=bool(argv[9]=='True'),
                whether_gen_fwd=bool(argv[10]=='True'))
            if data_en['answer_number'] == answer_en:
                correct_en += 1

            print(answer_en)
            print(data_en['answer_number'])


            task_instruction_small = task_instruction_set[argv[0]]
            prompt_small = prompt_set_1[argv[0]] + data_small['question'] + prompt_set_2[argv[0]]
            answer_small = Prompting(task_instruction_small, prompt_small,
                activate_keys_fwd_up_set=activate_keys_fwd_up_set,
                activate_keys_fwd_down_set=activate_keys_fwd_down_set,
                activate_keys_q_set=activate_keys_q_set,
                activate_keys_k_set=activate_keys_k_set,
                activate_keys_v_set=activate_keys_v_set, 
                under_layer=int(argv[1]), 
                gen_layer=int(argv[2]), 
                atten_number=int(argv[3]),
                ffn_number=int(argv[4]),
                whether_under=bool(argv[5]=='True'),
                whether_reason=bool(argv[6]=='True'),
                whether_gen=bool(argv[7]=='True'),
                whether_under_fwd=bool(argv[8]=='True'),
                whether_reason_fwd=bool(argv[9]=='True'),
                whether_gen_fwd=bool(argv[10]=='True'))
            if data_small['answer_number'] == answer_small:
                correct_small += 1

            print(answer_small)
            print(data_small['answer_number'])


            acc_en = correct_en / all_index
            acc_small = correct_small / all_index

            pbar.set_postfix(acc_en=f"{acc_en:.4f}", acc_small=f"{acc_small:.4f}")
            pbar.update(1)

    with open('./result/MGSM.txt', 'a', encoding='utf-8') as file:
        file.write('\n')
        file.write('Mistral ' + full_name[argv[0]]+' ' + str(argv[1])+' '+ str(argv[2])+' '+ str(argv[3])+' '+ str(argv[4])+' '+ str(argv[5])+' '+ str(argv[6])+' '+ str(argv[7]+' '+ str(argv[8]))+' '+ str(argv[9])+' '+ str(argv[10])+' '+ str(acc_en)+' '+str(acc_small))
        file.write('\n')


        



if __name__ == "__main__":
    main(sys.argv[1:])
