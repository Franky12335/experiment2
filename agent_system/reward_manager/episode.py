# Modified for the anonymous main training and evaluation release.
# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from verl import DataProto
import torch


class EpisodeRewardManager:
    """Assign the observed episode reward to the final response position."""

    def __init__(self, tokenizer, num_examine=0, normalize_by_length=False):
        self.tokenizer = tokenizer
        self.normalize_by_length = normalize_by_length

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch:
            reward = data.batch["rm_scores"]
        else:
            reward = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            for i in range(len(data)):
                item = data[i]
                prompt_length = item.batch["prompts"].shape[-1]
                response_length = item.batch["attention_mask"][prompt_length:].sum()
                score = item.non_tensor_batch["episode_rewards"]
                if self.normalize_by_length:
                    score = score / item.non_tensor_batch["episode_lengths"]
                reward[i, response_length - 1] = torch.tensor(
                    score, dtype=torch.float32, device=item.batch["prompts"].device,
                )
        if return_dict:
            return {"reward_tensor": reward, "reward_extra_info": {}}
        return reward
