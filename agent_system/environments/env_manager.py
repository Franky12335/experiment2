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

from functools import partial
from agent_system.environments.base import EnvironmentManagerBase
from agent_system.environments.search_manager import SearchEnvironmentManager


def make_envs(config):
    """Create the SearchQA training and evaluation environments."""
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    if config.env.env_name.lower() != "search":
        raise ValueError("The released experiment uses the SearchQA environment")
    from agent_system.environments.env_package.search import build_search_envs, search_projection

    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    training = build_search_envs(
        seed=config.env.seed, env_num=config.data.train_batch_size,
        group_n=group_n, is_train=True, env_config=config.env,
    )
    evaluation = build_search_envs(
        seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
        group_n=1, is_train=False, env_config=config.env,
    )
    projection = partial(search_projection)
    return (
        SearchEnvironmentManager(training, projection, config),
        SearchEnvironmentManager(evaluation, projection, config),
    )
