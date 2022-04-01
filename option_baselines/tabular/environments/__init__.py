from option_baselines.tabular.environments.fourrooms import Fourrooms
from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
)

register(
    id="Fourrooms-v0",
    entry_point="fourrooms:Fourrooms",
    max_episode_steps=100,
    reward_threshold=1,
)

