# AttackBench
AttackBench is a framework for testing and evaluating the robustness of large language models (LLMs) against various attack techniques. It implements multiple state-of-the-art attack methods to assess and help improve LLM safety.

interface: python
`python --attack --model_path --evaluation --dataset --num_GPU`
=> StrongReject Score


## Todo
AttackBench: evaluater
    - gpt evalution
    - Black list

- optimization：
    1. [x] add readme
    2. [x] run through 
    3. [x] add open ai evalution like [stronger](https://github.com/alexandrasouly/strongreject)



## Todo new
- run through 
    - [x] GCG
    - [x] GPTFuzzer
    - [x] ICA
    - [x] SURE
- add python interface
    - [x] unify the interface to call shell scripts
    - [x] attack method control
    - [x] model path control
    - [x] evaluation control   
    - [x]  GPU control
    - [x] dataset control 
- add strong reject
    - [x] GCG
    - [x] GPTFuzzer
    - [x] ICA
    - [x] SURE 
- Change output to ASR
    - [x] GCG
    - [x] GPTFuzzer
    - [x] ICA
    - [x] SURE 

## Overview

The main goal of this codebase is to:
1. Take a harmful question and target response
2. Try to find a sequence of tokens that can make the model generate that harmful response
3. Use this information to improve model safety

This is done by:
1. Starting with a random sequence of tokens
2. Using gradients to find which tokens to change
3. Testing different combinations of tokens
4. Evaluating if the attack was successful
5. Repeating until success or maximum steps reached


This project implements several LLM attack techniques:

- **GCG (Greedy Coordinate Gradient)**: An optimization-based attack method that crafts adversarial prompts to bypass LLM safety guardrails.
- **GPT Fuzzer**: A fuzzing-based approach to discover prompt vulnerabilities.
- **ICA (In-Context Attack)**: Uses in-context learning to construct harmful attacks.
- **SURE (Surrogate-based Uncertainty-aware Red-teaming)**: A surrogate model approach for red-teaming LLMs.

## Project Structure

```
AttackBench/
├── BOOST/                   # Core implementation of attack methods
│   ├── Attack_GCG/          # Greedy Coordinate Gradient attack implementation
│   ├── Attack_GPTFuzzer/    # GPT Fuzzer attack implementation
│   ├── Attack_ICA/          # In-Context Attack implementation
│   └── utils/               # Common utilities and helper functions
├── Dataset/                 # Harmful prompts and targets datasets
├── Experiments/             # Experiment runners for each attack type
│   ├── fuzzer_exp.py
│   ├── gcg_exp.py
│   ├── ica_exp.py
│   └── sure_exp.py
└── Scripts/                 # Shell scripts for running batch experiments
    └── run_GCG.sh           # Script for parallel GCG attacks using available GPUs
```

## Installation

```bash
# Clone the repository
git clone https://github.com/robinzixuan/AttackBench.git
cd AttackBench

# Install dependencies (inferred from imports)
conda env create -f environment.yml
conda activate jailbreak
```

## Usage

### Running GCG Attacks

The project supports running GCG attacks in parallel across multiple GPUs:

```bash
bash Scripts/run_GCG.sh
```

This script:
1. Searches for available GPUs with sufficient memory
2. Runs multiple attack instances in parallel
3. Saves results to the specified log directory

### Configuration

Edit the `Scripts/run_GCG.sh` file to configure:
- Target model (`MODEL_PATH`)
- Whether to add EOS tokens (`ADD_EOS`)
- Run index for logging purposes (`RUN_INDEX`)

### Datasets

The framework comes with several datasets:
- `harmful.csv`: Contains harmful prompts
- `harmful_targets.csv`: Contains targets for harmful prompts
- `Advbench.csv`: Adversarial benchmark dataset

## License

This project is licensed under the MIT License - see the LICENSE.txt file for details.

## Citation

If you use this code or find it helpful in your research, please consider citing our work.

## Disclaimer

This tool is intended for research purposes to improve LLM safety. The authors do not condone the use of this framework for malicious purposes or attacks on production systems.
