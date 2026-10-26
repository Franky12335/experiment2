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


from typing import Any, Dict, List, Tuple

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.prompts import SEARCH_TEMPLATE, SEARCH_TEMPLATE_NO_HIS
from agent_system.memory import SearchMemory


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        self._action_budget_hint = config.env.search.get("action_budget_hint", False)
        if type(self._action_budget_hint) is not bool:
            raise ValueError("Search action_budget_hint must be a boolean")
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs, *, batch_size: int) -> Tuple[Dict[str, Any], List[Dict]]:
        if len(kwargs) != batch_size:
            raise ValueError("Search task count must match the current rollout batch")
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy(),
            "retrieval_status": ["not_applicable"] * len(obs),
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy(),
            "retrieval_status": [info.get("retrieval_status", "api_error") for info in infos],
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
            info["projected_action"] = actions[i]

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            if self._action_budget_hint:
                completed = 0 if init else len(self.memory[i])
                remaining = max(int(self.config.env.max_steps) - completed, 0)
                if remaining == 1:
                    obs_i += (
                        "\n\nYou have 1 action remaining, including your final answer. "
                        "This is your final action. Give your best answer within <answer>...</answer> "
                        "using the question and the information already available. Do not search. "
                        "A search at this step will not be executed."
                    )
                elif remaining > 1:
                    obs_i += (
                        f"\n\nYou have {remaining} actions remaining, including your final answer. "
                        "Each search uses one action. Reserve the last action for your answer."
                    )
                else:
                    obs_i += "\n\nYou have 0 actions remaining. This episode has ended."
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

