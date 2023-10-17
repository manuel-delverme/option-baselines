from typing import Tuple

import torch
from torch import nn

from common.torch_layers import FakeOptimizer


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
