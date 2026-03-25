#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import os
os.environ["OPENAI_API_KEY"] = 'REDACTED_OPENAI_KEY'

ATTACK_TO_SCRIPT = {
    'gcg': 'run_GCG.sh',
    'fuzzer': 'run_GPTFuzzer_extra.sh',
    'ica': 'run_ICA.sh',
    'sure': 'run_SURE.sh',
    'reasoning': 'run_reasoning.sh',
}

def main():
    parser = argparse.ArgumentParser(description='Unified Python interface for running attack scripts.')
    parser.add_argument('--attack', required=True, choices=ATTACK_TO_SCRIPT.keys(), help='Attack method to run (gcg, fuzzer, ica, sure, reasoning)')
    parser.add_argument('--model_path', required=False, default='Qwen/Qwen3-4B-Thinking-2507', help='Model path to pass to the attack script')
    parser.add_argument('--evaluation', required=False, default='strongreject', help='Evaluation method to pass to the attack script (default or strongreject)')
    parser.add_argument('--num_tasks', type=int, default=50, help='Number of tasks to run in parallel (default: 3)')
    args = parser.parse_args()

    script_name = ATTACK_TO_SCRIPT[args.attack]
    script_path = os.path.join(os.path.dirname(__file__), script_name)

    # # Prepare environment variables
    # env = os.environ.copy()
    # env['MODEL_PATH'] = args.model_path
    # env['EVALUATION'] = args.evaluation

    print(f"[INFO] Running {script_name} with MODEL_PATH={args.model_path} and EVALUATION={args.evaluation}")
    try:
        result = subprocess.run([script_path, "--model_path", args.model_path, "--evaluation", args.evaluation, "--num_tasks", str(args.num_tasks)], check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {script_name} failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == '__main__':
    main() 