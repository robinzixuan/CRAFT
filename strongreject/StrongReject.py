import pandas as pd

pd.set_option("display.max_colwidth", None)
import getpass
import os

from strongreject.simple_jailbreak_runner import run_AIM_jailbreak, run_jb_on_dataset
from strongreject.strongreject_evaluator import run_autograder_on_dataset

if "OPENAI_API_KEY" in os.environ:
    api_key = os.environ["OPENAI_API_KEY"]
else:
    api_key = getpass.getpass("Enter your openai api key: ")
os.environ["OPENAI_API_KEY"] = api_key



# %%
data_path = "strongreject_dataset/strongreject_small_dataset.csv"
# full dataset:
# data_path = 'strongreject_dataset/strongreject_dataset.csv'
with open(data_path, "r") as file:
    df = pd.read_csv(file)


jb = run_AIM_jailbreak
results = run_jb_on_dataset(jb, df, model="gpt-3.5-turbo-0125")
results = run_autograder_on_dataset(results)
results.to_csv("strongreject/example_graded_answers.csv")


