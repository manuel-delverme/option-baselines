from typing import List, Tuple, Type

import gym
import torch.distributions
from stable_baselines3.common import policies
from stable_baselines3.common import torch_layers
from torch import nn

from option_baselines.common import constants


class Termination(policies.BaseModel):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            num_options: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        del action_space
        super().__init__(
            observation_space,
            action_space=None,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        self.num_options = num_options
        self.option_terminations = nn.ModuleList()

        for idx in range(num_options):
            q_net = torch_layers.create_mlp(features_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net, nn.Sigmoid())
            # self.add_module(f"termination_{idx}", q_net)
            self.option_terminations.append(q_net)

    def forward(self, observation: torch.Tensor, executing_option) -> Tuple[torch.Tensor, ...]:
        assert (torch.bitwise_or(executing_option < self.num_options, executing_option == constants.NO_OPTIONS)).all()
        assert (executing_option >= 0).all()
        features = self.extract_features(observation)
        termination_prob = torch.full((features.shape[0],), torch.nan)

        termination_prob[executing_option == constants.NO_OPTIONS] = 0.
        for option_idx, termination_net in enumerate(self.option_terminations):
            option_mask = executing_option == option_idx
            termination_prob[option_mask] = termination_net(features[option_mask]).squeeze()
        assert not torch.isnan(termination_prob).any()

        option_termination = torch.distributions.Bernoulli(termination_prob).sample()
        return option_termination.numpy().astype(dtype=bool), termination_prob
