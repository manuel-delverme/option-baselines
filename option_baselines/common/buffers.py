import dataclasses
from typing import Dict, Optional, Generator

import numpy as np
import torch
from gym import spaces
from stable_baselines3.common import buffers
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import VecNormalize


@dataclasses.dataclass
class OptionsRolloutBufferSamples:
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor  # TODO(Manuel): make plural like everything else
    advantages: torch.Tensor
    returns: torch.Tensor

    # Option stuff
    options: torch.Tensor
    option_values: torch.Tensor
    option_advantages: torch.Tensor
    option_returns: torch.Tensor
    option_log_probs: torch.Tensor
    # termination_probs: torch.Tensor


class DictOptionRolloutBufferSamples(OptionsRolloutBufferSamples):
    observations: type_aliases.TensorDict


class OptionRolloutBuffer(buffers.RolloutBuffer):
    def __init__(self, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Space, *args, **kwargs):
        super().__init__(buffer_size, observation_space, action_space)
        raise NotImplementedError("TODO: Same as DictionaryRolloutBuffer with no Dictionary")


class DictOptionRolloutBuffer(buffers.DictRolloutBuffer):
    def __init__(self, *args, **kwargs):
        # TODO: rename option to meta
        self._tensor_names = [
            "options",
            "option_log_probs",
            # "termination_probs",
            "option_values",
            "option_advantages",
            "option_returns",
        ]
        self.options = None
        self.option_values = None
        self.option_log_probs = None
        # self.termination_probs = None
        self.option_advantages = None
        self.option_returns = None
        assert all(hasattr(self, name) for name in self._tensor_names), "All tensor names must be defined in both _tensor_names and the class __init__, sorry"
        super().__init__(*args, **kwargs)

    def reset(self):
        self.options = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.option_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.option_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # self.termination_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.option_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictOptionRolloutBufferSamples:
        samples = super(DictOptionRolloutBuffer, self)._get_samples(batch_inds, env)
        return DictOptionRolloutBufferSamples(
            *samples,
            options=self.to_torch(self.options[batch_inds]),
            option_values=self.to_torch(self.option_values[batch_inds]),
            option_log_probs=self.to_torch(self.option_log_probs[batch_inds]),
            option_advantages=self.to_torch(self.option_advantages[batch_inds]),
            # termination_probs=self.to_torch(self.termination_probs[batch_inds]),
            option_returns=self.to_torch(self.option_returns[batch_inds]),
        )

    def add(
            self,
            obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: torch.Tensor,
            log_prob: torch.Tensor,
            option: Optional[np.ndarray] = None,
            option_value: Optional[torch.Tensor] = None,
            option_log_prob: Optional[torch.Tensor] = None,
            termination_prob: Optional[torch.Tensor] = None,
    ) -> None:
        option = np.full_like(self.options[self.pos], np.nan) if option is None else option
        option_value = torch.full(self.option_values[self.pos].shape, torch.nan) if option_value is None else option_value
        option_log_prob = (
            torch.full(self.option_log_probs[self.pos].shape, torch.nan) if option_log_prob is None else option_log_prob
        )
        termination_prob = (
            torch.full(self.termination_probs[self.pos].shape, torch.nan) if termination_prob is None else termination_prob
        )

        assert len(option_log_prob.shape) > 0, "option_log_prob can not be 0d"
        assert len(log_prob.shape) > 0, "log_prob2 can not be 0d"

        self.options[self.pos] = np.array(option).copy()
        self.option_values[self.pos] = option_value.clone().cpu().numpy().flatten()
        self.option_log_probs[self.pos] = option_log_prob.clone().cpu().numpy().squeeze()
        # self.termination_probs[self.pos] = termination_prob.clone().cpu().numpy().squeeze()
        super(DictOptionRolloutBuffer, self).add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None) -> Generator[OptionsRolloutBufferSamples, None, None]:
        if not self.generator_ready:
            for tensor in self._tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])

        yield from super(DictOptionRolloutBuffer, self).get(batch_size)

    def compute_returns_and_advantage(
            self, last_values: torch.Tensor, dones: np.ndarray, last_option_values: Optional[torch.Tensor] = None
    ) -> None:
        super(DictOptionRolloutBuffer, self).compute_returns_and_advantage(last_values, dones)

        if last_option_values is None:
            last_option_values = torch.zeros_like(last_values)
        del last_values
        self.option_advantages = np.full_like(self.advantages, np.nan)

        # TODO(Martin): halp, idk about the next
        last_option_values = last_option_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_option_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.option_values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.option_values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.option_advantages[step] = last_gae_lam

        for option_idx in np.unique(self.options):
            option_ind = self.options == option_idx
            self.option_advantages[option_ind] = self.returns[option_ind] - self.option_values[option_ind]
        self.option_returns = self.option_advantages + self.option_values
