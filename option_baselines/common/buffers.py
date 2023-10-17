import dataclasses
from typing import Dict, Optional, Generator, List

import numpy as np
import torch
from stable_baselines3.common import buffers
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import VecNormalize

from aoc.specs import OptionExecutionState


@dataclasses.dataclass
class OptionsRolloutBufferSamples:
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor  # TODO(Manuel): make plural like everything else
    advantages: torch.Tensor
    returns: torch.Tensor

    # Option stuff
    previous_options: torch.Tensor
    current_options: torch.Tensor
    meta_values: torch.Tensor
    option_advantages: torch.Tensor
    option_returns: torch.Tensor
    option_log_probs: torch.Tensor
    option_available: torch.Tensor
    # termination_probs: torch.Tensor


class DictOptionRolloutBufferSamples(OptionsRolloutBufferSamples):
    observations: type_aliases.TensorDict


class DictOptionRolloutBuffer(buffers.DictRolloutBuffer):
    def __init__(self, *args, **kwargs):
        # TODO: rename option to meta
        self._meta_tensor_names = [
            "current_options",
            "previous_options",
            "option_log_probs",
            "meta_values",
            "option_advantages",
            "option_returns",
            "option_available",
        ]
        self.previous_options = None
        self.current_options = None
        self.meta_values = None
        self.option_log_probs = None
        # self.termination_probs = None
        self.option_advantages = None
        self.option_returns = None
        self.option_available = None

        self.num_options = kwargs.pop("num_options")

        assert all(hasattr(self, name) for name in self._meta_tensor_names), (
            "All tensor names must be defined in both _meta_tensor_names and the class __init__, sorry")
        super().__init__(*args, **kwargs)

    def reset(self):
        self.current_options = np.empty((self.buffer_size, self.n_envs), dtype=np.int64)
        self.previous_options = np.empty((self.buffer_size, self.n_envs), dtype=np.int64)
        self.meta_values = np.empty((self.buffer_size, self.n_envs), dtype=np.float32)
        self.option_log_probs = np.empty((self.buffer_size, self.n_envs), dtype=np.float32)
        self.option_advantages = np.empty((self.buffer_size, self.n_envs), dtype=np.float32)
        self.option_available = np.empty((self.buffer_size, self.n_envs, self.num_options), dtype=bool)
        super().reset()

    def _get_samples(self, batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None) -> DictOptionRolloutBufferSamples:
        samples = super(DictOptionRolloutBuffer, self)._get_samples(batch_inds, env)
        return DictOptionRolloutBufferSamples(
            *samples,
            previous_options=self.to_torch(self.previous_options[batch_inds]).squeeze(1),
            current_options=self.to_torch(self.current_options[batch_inds]).squeeze(1),
            meta_values=self.to_torch(self.meta_values[batch_inds]).squeeze(1),
            option_log_probs=self.to_torch(self.option_log_probs[batch_inds]).squeeze(1),
            option_advantages=self.to_torch(self.option_advantages[batch_inds].squeeze(1)),
            option_returns=self.to_torch(self.option_returns[batch_inds]).squeeze(1),
            option_available=self.to_torch(self.option_available[batch_inds]),
        )

    def add(
            self,
            obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: torch.Tensor,
            log_prob: torch.Tensor,
            previous_option: Optional[torch.Tensor] = None,
            current_option: Optional[torch.Tensor] = None,
            meta_value: Optional[torch.Tensor] = None,
            option_log_prob: Optional[torch.Tensor] = None,
            option_available: Optional[torch.Tensor] = None,
    ) -> None:
        assert option_available is not None, "This buffer only supports HRL and requires state"
        assert len(option_log_prob.shape) > 0, "option_log_prob can not be 0d"
        assert len(log_prob.shape) > 0, "log_prob2 can not be 0d"
        assert (current_option >= 0).all()
        assert torch.bitwise_or(previous_option >= 0, torch.isnan(previous_option)).all()

        self.previous_options[self.pos] = previous_option.cpu().numpy().copy()
        self.current_options[self.pos] = current_option.cpu().numpy().copy()

        self.meta_values[self.pos] = meta_value.clone().cpu().numpy().flatten()
        self.option_log_probs[self.pos] = option_log_prob.clone().cpu().numpy().squeeze()
        self.option_available[self.pos] = option_available.clone().cpu().numpy().squeeze()
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None) -> Generator[OptionsRolloutBufferSamples, None, None]:
        if not self.generator_ready:
            for tensor in self._meta_tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])

        yield from super().get(batch_size)

    def compute_returns_and_advantage(
            self, last_values: torch.Tensor, dones: np.ndarray, last_value_upon_arrival: Optional[torch.Tensor] = None,
            option_termination_probs: Optional[torch.Tensor] = None,
    ) -> None:
        super().compute_returns_and_advantage(last_values, dones)

        self.option_advantages = np.full_like(self.advantages, np.nan)
        last_value_upon_arrival = last_value_upon_arrival.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_value_upon_arrival
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.meta_values[step + 1]

            future_value = next_values * next_non_terminal - self.meta_values[step]

            delta = self.rewards[step] + self.gamma * future_value
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.option_advantages[step] = last_gae_lam

        self.option_returns = self.option_advantages + self.meta_values
