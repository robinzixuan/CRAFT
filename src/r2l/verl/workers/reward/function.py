# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
import time
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict, NamedTuple

import torch
from transformers import PreTrainedTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
from strongreject.strongreject_evaluator import autograde_response




from ...protocol import DataProto
from .config import RewardConfig
import torch.nn as nn
import torch.nn.functional as F


class EvalResult(NamedTuple):
    success: bool
    score: float


class RewardInput(TypedDict):
    response: str
    p_text: float
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_ids: torch.Tensor
    response_mask: torch.Tensor
    response_length: int
    h_mean: Optional[torch.Tensor]
    config: RewardConfig
    

def _sanitize_text(text: str) -> str:
    """Remove null bytes and non-printable control chars that break JSON serialization."""
    # Remove null bytes and ASCII control chars (except tab/newline/CR which are valid JSON)
    return "".join(c for c in text if c == "\t" or c == "\n" or c == "\r" or (ord(c) >= 32 and c != "\x7f"))


def evaluate_generation_strongreject(question, generation, max_retries=8):
    """Call autograde_response with exponential backoff on rate limit (429) errors.
    Returns 0.0 for bad-request errors (e.g. unparseable text with control chars).
    """
    import openai
    import re
    question = _sanitize_text(question)
    generation = _sanitize_text(generation)
    delay = 1.0
    for attempt in range(max_retries):
        try:
            result = autograde_response(question, generation)
            success = result.score_zero_to_one is not None and result.score_zero_to_one > 0
            score = result.score_zero_to_one if result.score_zero_to_one is not None else 0.0
            return EvalResult(success, score)[1]
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = delay + random.uniform(0, 0.5)
            m = re.search(r"try again in (\d+(?:\.\d+)?)s", str(e))
            if m:
                wait = max(wait, float(m.group(1)) + random.uniform(0, 0.5))
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
        except openai.BadRequestError:
            # Unrecoverable bad input (e.g. residual control chars after sanitization).
            # Return neutral score so training continues.
            return 0.0

class RewardScore(TypedDict):
    overall: float
    R_latent: Optional[float]
    R_text: Optional[float]
    R_cons: Optional[float]








SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class SequentialFunctionRewardManagerMixin:
    reward_fn: SequentialRewardFunction

    def compute_reward_sequential(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManagerMixin:
    reward_fn: BatchRewardFunction

    def compute_reward_batch(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        response_mask = data.batch["response_mask"]
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        prompt_len = input_ids.shape[1] - response_ids.shape[1]
        h_mean_available = "h_mean" in data.batch.keys()

        # decode all texts first (CPU, fast)
        samples = []
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())
            response_str = self.tokenizer.decode(
                response_ids[i][:cur_response_length], skip_special_tokens=self.config.skip_special_tokens
            )
            prompt_str = self.tokenizer.decode(input_ids[i][:prompt_len], skip_special_tokens=True)
            samples.append((i, cur_response_length, prompt_str, response_str))

        # fire all OpenAI API calls in parallel (I/O-bound → threads)
        p_text_scores = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=min(len(samples), 8)) as executor:
            futures = {
                executor.submit(evaluate_generation_strongreject, prompt_str, response_str): i
                for i, _, prompt_str, response_str in samples
            }
            for future in as_completed(futures):
                idx = futures[future]
                p_text_scores[idx] = future.result()

        reward_inputs = []
        for i, cur_response_length, _, response_str in samples:
            reward_inputs.append(
                {
                    "response": response_str,
                    "p_text": p_text_scores[i],
                    'input_ids': input_ids[i],
                    'attention_mask': attention_mask[i],
                    'response_ids': response_ids[i],
                    'response_mask': response_mask[i],
                    "response_length": cur_response_length,
                    "h_mean": data.batch["h_mean"][i] if h_mean_available else None,
                    "config": self.config,
                }
            )

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class AutoRewardManager(BatchFunctionRewardManagerMixin, SequentialFunctionRewardManagerMixin):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        self.config = config
        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        reward_name = getattr(module, "REWARD_NAME", "unknown")
        reward_type = getattr(module, "REWARD_TYPE", "batch")
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_type = reward_type
        self.config = config
        self.tokenizer = tokenizer


        


    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        if self.reward_type == "batch":
            return self.compute_reward_batch(data)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(data)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")
