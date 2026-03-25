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
"""
Reward config
"""

from dataclasses import dataclass, field
from typing import Optional

from ...utils.py_functional import get_abs_path


@dataclass
class RewardConfig:
    reward_function_kwargs: dict = field(default_factory=dict)
    reward_type: str = "batch"
    reward_function: Optional[str] = None
    skip_special_tokens: bool = True
    num_cpus: int = 1
    num_gpus: int = 1  # Number of GPUs for reward manager (required for batch type with vLLM)
    # Model configuration for vLLM
    model_path: Optional[str] = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.6
    trust_remote_code: bool = False
    projection_ckpt_path: Optional[str] = None
    safety_ckpt_path: Optional[str] = None
    projection_dim: int = 512
    # below are auto keys
    reward_function_name: Optional[str] = field(default=None, init=False)

    def post_init(self):
        if self.reward_function is not None:  # support custom reward function, e.g., ./math.py:main
            if ":" not in self.reward_function:
                self.reward_function_name = "main"
            else:
                self.reward_function, self.reward_function_name = self.reward_function.rsplit(":", maxsplit=1)

            self.reward_function = get_abs_path(self.reward_function, prompt="Reward function")
