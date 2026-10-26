# Modified for the anonymous main training and evaluation release.
"""Registers the internal gym envs."""

from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.registration import register

register(
    id="search",
    entry_point="agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.env:SearchEnv",
)
