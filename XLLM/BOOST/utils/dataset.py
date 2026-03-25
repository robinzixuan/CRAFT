import random
import pandas as pd
random.seed(42)


def get_dataset(subset, harmful_dataset, harmless_dataset, jailbreak_templates):
    if subset['harmful']:
        base_dataset = harmful_dataset
    else:
        base_dataset = harmless_dataset
        
    if 'use_jailbreak_template' in subset and subset['use_jailbreak_template']:
        if not subset['different_jb_templates']:
            if subset['random_jailbreak_templates']:
                jb_template = random.choice(jailbreak_templates)
            else:
                jb_template = jailbreak_templates[subset['use_template_index']]
            sub_dataset = [jb_template.replace("[INSERT PROMPT HERE]", s) for s in base_dataset]
        else:
            sub_dataset = []
            for i in range(len(base_dataset)):
                jb_template = random.choice(jailbreak_templates)
                sub_dataset.append(jb_template.replace("[INSERT PROMPT HERE]", base_dataset[i]))
    
    elif 'use_success_jb_template' in subset and subset['use_success_jb_template']:
        sub_dataset = []
        for i in range(len(base_dataset)):
            try:
                success_jb_template_path = 'datasets/jb_prompts/' + str(i) + '.csv'
                success_jb_template = pd.read_csv(success_jb_template_path)['prompt'].tolist()
                # random choose one
                jb_template = random.choice(success_jb_template)
                sub_dataset.append(jb_template.replace("[INSERT PROMPT HERE]", base_dataset[i]))
            except Exception as e:
                print(f"Error reading success jailbreak template for index {i}")
                print(e)
                print("Skipping this index")
                
    elif 'use_gcg' in subset and subset['use_gcg']:
        sub_dataset = []
        for i in range(len(base_dataset)):
            try:
                gcg_template_path = 'datasets/gcg/' + str(i) + '.csv'
                gcg_template = pd.read_csv(gcg_template_path)['optim_prompts'].tolist()
                # random choose one
                gcg_prompt = random.choice(gcg_template)
                sub_dataset.append(gcg_prompt)
            except Exception as e:
                print(f"Error reading gcg template for index {i}")
                print(e)
                print("Skipping this index")
    
    elif 'add_eos' in subset and subset['add_eos']:
        eos_num = subset['eos_num']
        sub_dataset = [s + '</s>'*eos_num for s in base_dataset]
        
    else:
        sub_dataset = base_dataset
        
    return sub_dataset