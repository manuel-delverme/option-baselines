from typing import List, Tuple, Type, Dict, Any
from option_baselines.common.torch_layers import FakeOptimizer

import gym
import torch.distributions
from stable_baselines3.common import policies
from stable_baselines3.common import torch_layers
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from option_baselines.common import constants


class HardcodedTermination(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.forced_prob = kwargs.pop("forced_prob")
        self.optimizer = FakeOptimizer()

    def forward(self, observation: torch.Tensor, executing_option) -> Tuple[torch.Tensor, ...]:
        termination_prob = torch.full(executing_option.shape, self.forced_prob)
        option_termination = torch.distributions.Bernoulli(termination_prob).sample()
        return option_termination.numpy().astype(dtype=bool), termination_prob

    def forward_offpolicy(self, observation: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def _update_learning_rate(self, current_progress_remaining) -> None:
        raise NotImplementedError


class Termination(policies.BaseModel):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor_class: Type[CombinedExtractor],
            optimizer_class: Type[torch.optim.Optimizer],
            optimizer_kwargs: Dict[str, Any],
            features_extractor_kwargs: dict,
            num_options: int,
            lr_schedule: Schedule,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        features_extractor = features_extractor_class(
            observation_space,
            **features_extractor_kwargs)
        super().__init__(
            observation_space,
            action_space=None,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        self.num_options = num_options
        self.option_terminations = nn.ModuleList()

        for idx in range(num_options):
            q_net = torch_layers.create_mlp(self.features_extractor.features_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net, nn.Sigmoid())
            self.option_terminations.append(q_net)

        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_class = optimizer_class

        # This should be in build, but I'd like to deprecate build

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, observation: torch.Tensor, executing_option) -> Tuple[torch.Tensor, ...]:
        assert (torch.bitwise_or(executing_option < self.num_options, executing_option == constants.NO_OPTIONS)).all()
        assert (executing_option >= 0).all()
        features = self.extract_features(observation)
        termination_prob = torch.full((features.shape[0],), float("nan"))

        termination_prob[executing_option == constants.NO_OPTIONS] = 0.
        for option_idx, termination_net in enumerate(self.option_terminations):
            option_mask = executing_option == option_idx
            termination_prob[option_mask] = termination_net(features[option_mask]).squeeze()
        assert not torch.isnan(termination_prob).any()

        option_termination = torch.distributions.Bernoulli(termination_prob).sample()
        return option_termination.numpy().astype(dtype=bool), termination_prob

    def forward_offpolicy(self, observation: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        features = self.extract_features(observation)
        termination_prob = torch.full((features.shape[0], len(self.option_terminations)), float("nan"))
        option_termination = torch.full((features.shape[0], len(self.option_terminations)), float("nan"))

        for option_idx, termination_net in enumerate(self.option_terminations):
            termination_prob[:, option_idx] = termination_net(features).squeeze()
            option_termination[:, option_idx] = torch.distributions.Bernoulli(termination_prob[:, option_idx]).sample()
        return option_termination.numpy().astype(dtype=bool), termination_prob

    def _update_learning_rate(self, current_progress_remaining) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr_schedule(current_progress_remaining)
