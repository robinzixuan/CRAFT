import jailbreakbench as jbb
import datasets
import os
from datasets import load_dataset

data = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split='harmful')

llm = jbb.LLMvLLM(model_name="qwen-3-4b-thinking")



for i, row in enumerate(data):
    prompts = [row['Goal']]
    behavior = row['Behavior']
    print(f"Processing {i+1}/{len(data)}: {behavior}")
    responses = llm.query(prompts=prompts, behavior=behavior)
    print(f"Response: {responses}")
    break
