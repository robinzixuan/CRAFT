import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # for debugging
import pandas as pd
import argparse
from BOOST.Attack_GCG.gcg import GCG
import csv
from BOOST.utils.templates import get_eos

def gcg_attack(args):

    question = pd.read_csv(args.harmful_dataset)['text'].tolist()[args.index]
    target = pd.read_csv(args.targets_dataset)['target'].tolist()[args.index]

    if args.add_eos:
        eos_token = get_eos(args.model_path)
        question = question + eos_token * args.eos_num
    args.question = question
    print("args.index: ", args.index)
    print("The question is: ", question)
    print("The target is: ", target)
    
    gcg = GCG(args)
    optim_prompts, steps, scores = gcg.run(target)
    
    # save the optim prompts into a csv file
    save_path = f'./Results/{args.model_path}/GCG-{args.run_index}/{args.index}.csv'
    if args.add_eos:
        save_path = f'./Results/{args.model_path}/GCG_eos-{args.run_index}/{args.index}.csv'
    # Add evaluation method as a folder
    evaluation = getattr(args, 'evaluation', 'default')
    save_path = save_path.replace(f'/{args.index}.csv', f'/{evaluation}/{args.index}.csv')
        
    # check if the directory exists
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        #write the column name
        if evaluation == 'strongreject':
            writer.writerow(['optim_prompts', 'steps', 'scores'])
            for prompt, step, score in zip(optim_prompts, steps, scores):
                writer.writerow([prompt, step, score])
        else:
            writer.writerow(['optim_prompts', 'steps'])
            for prompt, step in zip(optim_prompts, steps):
                writer.writerow([prompt, step])
    
    print("The optim prompts are saved.")