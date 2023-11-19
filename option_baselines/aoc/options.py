import collections
import copy
import logging
import os
import sys
import time
import uuid
from typing import List, Tuple, Type, Dict, Any, Optional
from typing import Union, Mapping

import gym
import numpy as np
import torch
import torch.distributions
import torch.nn
import torch.nn.functional
from stable_baselines3.common import distributions, policies
from stable_baselines3.common import torch_layers
from stable_baselines3.common import utils as sb3_utils
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.vec_env import VecEnv
from torch import nn
from torch.nn import functional as F

from option_baselines.aoc.specs import OptionExecutionState
from option_baselines.common import buffers
from option_baselines.common import constants


def setup_hrl(option_policy_class, meta_policy_class, initiation_class, termination_class,
              option_policy_kwargs, meta_policy_kwargs, initiation_kwargs, termination_kwargs,
              action_space, device, num_options, observation_space):
    option_policies = torch.nn.ModuleList()  # Policy is a keyword for stable-baselines.
    for _ in range(num_options):
        option_policies.append(option_policy_class(observation_space, action_space, **option_policy_kwargs).to(device))
    option_space = gym.spaces.Discrete(len(option_policies))
    meta_policy = meta_policy_class(observation_space, option_space, **meta_policy_kwargs).to(device)
    initialization = initiation_class(observation_space, **initiation_kwargs, ).to(device)
    terminations = termination_class(observation_space, num_options=num_options, **termination_kwargs).to(device)
    # Policy is a keyword for stable-baselines.
    # Does not really reflect the current hierarchical structure but it's hardcoded in the library
    # a better name would be self.parameters
    policy_kwargs = {
        "option_policies": option_policies,
        "meta_policy": meta_policy,
        "initialization": initialization,
        "termination": terminations,
    }
    return policy_kwargs


class MetaAC(policies.MultiInputActorCriticPolicy):
    pass


class SimpleAC(policies.MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        random = os.urandom(16)
        curr_time = time.time()
        current_time = bytes(curr_time.hex(), encoding="utf-8")
        bytes_seq = random + current_time
        self.option_hash = uuid.UUID(bytes=bytes_seq[:16], version=4).hex[:8]
        curr_time_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(curr_time))
        logging.info(f">>>>>>> Option hash: current_time: {curr_time_iso}: {current_time}, random: {random} => {self.option_hash}")


class Termination(policies.BaseModel):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor_class: Type[torch_layers.CombinedExtractor],
            optimizer_class: Type[torch.optim.Optimizer],
            optimizer_kwargs: Dict[str, Any],
            features_extractor_kwargs: Optional[dict],
            num_options: int,
            lr_schedule: Schedule,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        features_extractor = features_extractor_class(observation_space, **features_extractor_kwargs)
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

        # This should be in build, but I'd like to deprecate build
        self.optimizer = optimizer_class(self.parameters(), lr=lr_schedule(1), **optimizer_kwargs)

    def forward(self, observation: torch.Tensor, executing_option: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        features = self.extract_features(observation)
        termination_prob = torch.full((features.shape[0],), float("nan"), device=features.device, dtype=torch.float32)

        termination_prob[executing_option == constants.NO_OPTIONS] = 0.
        for option_idx, termination_net in enumerate(self.option_terminations):
            option_mask = executing_option == option_idx
            termination_prob[option_mask] = termination_net(features[option_mask]).squeeze()
            # termination_prob[option_mask] = torch.clamp(termination_prob[option_mask], 0., 1.)
        assert not torch.isnan(termination_prob).any()

        option_termination = torch.distributions.Bernoulli(termination_prob).sample()
        return option_termination.to(dtype=torch.bool), termination_prob

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


class Initiation(Termination):
    def forward(self, observation: torch.Tensor, executing_option) -> Tuple[torch.Tensor, ...]:
        is_unavailable, unavailable_prob = self.forward_offpolicy(observation)
        is_unavailable = is_unavailable[:, executing_option]
        unavailable_prob = unavailable_prob[:, executing_option]
        return is_unavailable, unavailable_prob

    def forward_offpolicy(self, observation: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        is_unavailable, unavailable_prob = super().forward_offpolicy(observation)
        no_actions = is_unavailable.all(dim=1)
        is_unavailable[no_actions] = False
        return is_unavailable, unavailable_prob


class UnusedFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space):
        super().__init__(observation_space, features_dim=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PPOOC(OnPolicyAlgorithm):
    def __init__(
            self,
            meta_policy_class: Type[policies.ActorCriticPolicy],
            option_policy_class: Type[policies.ActorCriticPolicy],
            initialization_class: Type[Initiation],
            termination_class: Type[Termination],

            meta_policy_kwargs: Dict[str, Any],
            option_policy_kwargs: Dict[str, Any],
            termination_kwargs: Dict[str, Any],
            initialization_kwargs: Dict[str, Any],

            env: Union[GymEnv, str],
            num_options: int,
            batch_size: int,
            n_epochs: int,
            clip_range: float,
            initial_ent_coef: float,
            final_ent_coef: float,
            n_steps: int = 5,
            gamma: float = 0.99,
            gae_lambda: float = 1.0,
            meta_ent_coef: float = 0.0,
            term_coef=0.01,
            vf_coef: float = 0.5,
            switching_margin: float = 0.0,
            max_grad_norm: float = 0.5,
            sde_sample_freq: int = -1,
            normalize_advantage: bool = False,
            offpolicy_learning: bool = True,
            tensorboard_log: Optional[str] = None,

            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[torch.device, str] = "auto",

    ):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range

        self.term_coef = term_coef
        self.meta_ent_coef = meta_ent_coef
        self.switching_margin = switching_margin
        self.normalize_advantage = normalize_advantage
        self.offpolicy_learning = offpolicy_learning
        self.num_options = num_options

        self.entropy_scheduler = sb3_utils.get_linear_fn(
            start=initial_ent_coef, end=initial_ent_coef / final_ent_coef, end_fraction=0.5)

        assert isinstance(env.observation_space, gym.spaces.Dict), "Non-dict observation spaces were simplified away"
        self.buffer_cls = buffers.DictOptionRolloutBuffer

        policy_kwargs = {
            "option_policy_class": option_policy_class,
            "meta_policy_class": meta_policy_class,
            "initialization_class": initialization_class,
            "termination_class": termination_class,

            "option_policy_kwargs": option_policy_kwargs,
            "meta_policy_kwargs": meta_policy_kwargs,
            "initialization_kwargs": initialization_kwargs,
            "termination_kwargs": termination_kwargs,

            "device": device,
            "num_options": num_options,
            # "optimizer_class": optimizer_class,
            # "optimizer_kwargs": optimizer_kwargs,
        }

        super().__init__(
            policy=HierarchicalPolicy,
            policy_kwargs=policy_kwargs,
            env=env,
            learning_rate=31337.,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=initial_ent_coef,
            vf_coef=vf_coef,
            use_sde=False,
            max_grad_norm=max_grad_norm,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=True,
            supported_action_spaces=(gym.spaces.Discrete,),
        )
        self.set_random_seed(self.seed)
        self.rollout_buffer: buffers.DictOptionRolloutBuffer = self.buffer_cls(
            # TODO(Manuel) actually this should be OptionRolloutBuffer
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            num_options=num_options,
        )

        # TODO: remove this hack
        # self.eval_executing_option_history = []
        self._last_rollout_states = None

    def _setup_learn(self, *args, **kwargs):
        retr = super()._setup_learn(*args, **kwargs)
        is_available, _ = self.policy.initialization.forward_offpolicy(self._last_obs)
        self._last_rollout_states = OptionExecutionState(
            torch.full(size=(self.n_envs,), fill_value=constants.NO_OPTIONS),
            option_is_available=is_available)
        return retr

    @property
    def _ent_coef(self):
        return self.entropy_scheduler(self._current_progress_remaining)

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: buffers.DictOptionRolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        assert self._last_rollout_states is not None, "No previous rollout states were provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        dones = self._last_episode_starts
        while n_steps < n_rollout_steps:
            with torch.no_grad():

                obs_tensor = sb3_utils.obs_as_tensor(self._last_obs, self.device)
                dones = torch.tensor(self._last_episode_starts, dtype=torch.bool)

                retr = self.policy(obs_tensor, self._last_rollout_states, dones)
                state, actions, values, log_probs, meta_values, option_log_probs, termination_probs = retr
                if actions.max().item() >= self.action_space.n:
                    raise ValueError("Actions not in action space")

            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                current_option=state.executing_option,
                previous_option=self._last_rollout_states.executing_option,
                meta_value=meta_values,
                option_log_prob=option_log_probs,
                option_available=state.option_is_available,
            )
            del values

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_rollout_states = copy.deepcopy(state)

        with torch.no_grad():
            # Compute value for the last time-step
            new_obs = sb3_utils.obs_as_tensor(new_obs, self.device)
            values, meta_value = self.policy.predict_values(new_obs, state.executing_option)

            # beta(S',O) * V(S')
            termination_value = torch.einsum("b,b->b", termination_probs, meta_value)

            # (1 - beta(S',O)) * Q(S', O)
            continuation_value = torch.einsum("b,b->b", (1 - termination_probs), values)

            value_upon_arrival = termination_value + continuation_value
            value_upon_arrival[dones] = 0

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones,
                                                     last_value_upon_arrival=value_upon_arrival,
                                                     option_termination_probs=termination_probs)

        callback.on_rollout_end()

        return True

    def _update_learning_rate(self, optimizers: List[torch.optim.Optimizer]) -> None:
        return

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        entropy_losses = []
        pg_losses, action_value_losses = [], []
        clip_fractions = []
        continue_training = True

        self.n_batches = max((self.n_epochs * self.rollout_buffer.actions.size) // self.batch_size, 1)

        assert self.n_epochs > 0, "Not enough data to train on"
        # train for n_epochs epochs
        for epoch in range(self.n_batches):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = actions.long().flatten()

                previous_options = rollout_data.previous_options
                # current_options = rollout_data.current_options
                # states = rollout_data.states
                states = OptionExecutionState(
                    executing_option=rollout_data.current_options,
                    option_is_available=rollout_data.option_available,
                )

                observations = rollout_data.observations

                (meta_values, meta_log_prob, meta_entropy
                 ), (action_values, action_log_prob, entropy,) = self.policy.evaluate_actions(
                    observations, states, actions)

                _, termination_probs = self.policy.terminations(observations, previous_options)
                action_values, meta_values = action_values.flatten(), meta_values.flatten()

                # Normalize advantage (not present in the original implementation)
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                meta_advantages = rollout_data.option_advantages

                if self.normalize_advantage:
                    meta_advantages = (meta_advantages - meta_advantages.mean()) / (meta_advantages.std() + 1e-8)

                ratio = torch.exp(action_log_prob - rollout_data.old_log_prob)
                # torch.exp(action_log_prob - rollout_data.old_log_prob) * advantages
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # policy_loss = -(advantages * action_log_prob).mean()
                meta_log_prob = torch.clamp(meta_log_prob, -10, 10)

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.offpolicy_learning:
                    raise NotImplementedError
                else:
                    controllable_meta_advantages = torch.einsum("b,b->b", termination_probs.detach(), meta_advantages)
                    meta_entropy = torch.einsum("b,b->b", termination_probs.detach(), meta_entropy)
                    meta_policy_loss = -(controllable_meta_advantages * meta_log_prob).mean()

                value_loss = F.mse_loss(rollout_data.returns, action_values)
                action_value_losses.append(value_loss.item())

                if self.offpolicy_learning:
                    raise NotImplementedError
                else:
                    value_error = rollout_data.option_returns - meta_values
                    weighted_value_error = torch.einsum("b,b->b", termination_probs.detach(), value_error)
                    meta_value_loss = weighted_value_error.pow(2).mean()

                # Entropy loss favor exploration, approximate entropy when no analytical form
                meta_entropies = -meta_log_prob if meta_entropy is None else meta_entropy
                entropies = -action_log_prob if entropy is None else entropy

                meta_entropy_loss = -torch.mean(meta_entropies * self.meta_ent_coef)
                entropy_loss = -torch.mean(entropies * self._ent_coef)

                # Option loss
                loss = self.loss_fn(locals(), globals())

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = action_log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()

                # Clip grad norm
                grad_means = {}
                gradz = []
                with torch.no_grad():
                    for k, v in self.policy.named_children():
                        child_grads = [(g.detach().abs().mean(), f"{k}.{kj}") for kj, g in v.named_parameters() if
                                       g.grad is not None]
                        if not child_grads:
                            continue

                        gradz.extend(child_grads)
                        grads = [g.item() for g, _ in child_grads]
                        grad_means[k] = torch.mean(torch.tensor(grads))

                    if any([torch.isnan(v) for v in grad_means.values()]):
                        print("grad_mean is nan")
                        raise ValueError("grad_mean is nan")

                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                assert not list(self.policy.parameters(recurse=False)), "Found parameters not in any child module"

                self.policy.optimizer.step()

        self._n_updates += self.n_epochs
        explained_var = sb3_utils.explained_variance(self.rollout_buffer.values.flatten(),
                                                     self.rollout_buffer.returns.flatten())

        self._n_updates += 1

        meta_entropy = meta_entropy.mean().item()
        option_entropy = entropy.mean().item()
        grad_means = {k: v.item() for k, v in grad_means.items()}

        metrics = {
            "train/policy_gradient_loss": np.mean(pg_losses),
            "train/action_value_loss": np.mean(action_value_losses),
            "train/approx_kl": np.mean(approx_kl_divs),
            "train/clip_fraction": np.mean(clip_fractions),
            "option/entropy_coeff": self._ent_coef,
            "option/entropy": option_entropy,
            "train/grad_norm": grad_norm,
            "train/policy_loss": policy_loss.item(),
            "train/value_loss": value_loss.item(),
            "train/advantages": advantages.mean().item(),
            "meta_train/entropy": meta_entropy,
            "meta_train/policy_loss": meta_policy_loss.item(),
            "meta_train/value_loss": meta_value_loss.item(),
            "meta_train/advantages": meta_advantages.mean().item(),
            **{"grad_mean/" + k: v for k, v in grad_means.items()},
        }
        if hasattr(self.policy, "log_std"):
            metrics["train/std"] = torch.exp(self.policy.log_std).mean().item()
        self.logger.log(metrics, commit=False)

    def loss_fn(self, locals_, globals_) -> torch.Tensor:
        loss = torch.tensor(0, dtype=torch.float32)
        loss += locals_["policy_loss"]
        loss += locals_["meta_policy_loss"]
        loss += self.vf_coef * locals_["value_loss"]
        loss += self.vf_coef * locals_["meta_value_loss"]
        loss += locals_["entropy_loss"]
        loss += locals_["meta_entropy_loss"]
        loss += self.termination_loss(locals_, globals_)
        loss += self.auxiliary_loss(locals_, globals_)
        # TODO: remove termination and auxiliary loss function
        return loss

    def termination_loss(self, locals_, _globals):
        meta_advantages = locals_["meta_advantages"]
        termination_probs = locals_["termination_probs"]
        termination_logprob = torch.log(termination_probs)

        value_to_continue = meta_advantages.detach() + self.switching_margin
        margin_loss = (value_to_continue * termination_logprob).mean()
        termination_loss = termination_probs.mean()
        termination_mean = termination_probs.mean().item()

        # There is no way to fix it, reintroduce KL regularization to some baseline temporal extension

        self.logger.log({
            "train/margin_loss": margin_loss.item(),
            "train/termination_loss": termination_loss.item(),
            "train/termination_logprob": termination_logprob.mean().item(),
            "train/value_to_continue": value_to_continue.mean().item(),
            "train/termination_mean": termination_mean,
        })

        loss = termination_loss * self.term_coef + margin_loss
        return loss

    def auxiliary_loss(self, locals_, globals_):
        return 0.0


class KLTermination(Termination):
    def __init__(self, *args, **kwargs):
        self.target_termination_prob = kwargs.pop("target_termination_prob")
        self.termination_kl_weight = kwargs.pop("termination_kl_weight")
        super().__init__(*args, **kwargs)

    def termination_loss(self, locals_, globals_):
        termination_metrics = {}

        loss = super().termination_loss(locals_, globals_)
        termination_probs = locals_["termination_probs"]
        first_option = locals_["previous_options"] == constants.NO_OPTIONS

        mask = torch.logical_not(first_option)

        executing_option = locals_["previous_options"]

        for option_idx in executing_option.unique().tolist():
            if option_idx == constants.NO_OPTIONS:
                continue
            termination_prob = termination_probs[executing_option == option_idx].mean().item()
            termination_metrics[f"termination/prob_{option_idx}"] = termination_prob

        if self.termination_kl_weight == 0. or not mask.any():
            termination_kl_loss = torch.tensor(0.)
        else:
            termination_metrics["termination/probs"] = termination_probs[mask].mean().item()

            termination_kl_loss = self.kl_dist(self.target_termination_prob, termination_probs[mask])

            # Extreme termination with hard-coded value since the KL divergence is infinite and the gradient is NaN
            termination_kl_loss[torch.isinf(termination_kl_loss)] = 0.
            termination_kl_loss = termination_kl_loss.mean()

            termination_metrics["termination/kl"] = termination_kl_loss.item()

        self.logger.log(termination_metrics, commit=False)
        return loss + termination_kl_loss * self.termination_kl_weight

    def kl_dist(self, p, q):
        t1 = p * (p / q).log()
        t2 = (1 - p) * ((1 - p) / (1 - q)).log()
        t1[p == 0] = 0  # Avoid infinity / NaNs
        t2[p == 1] = 0
        termination_kl_loss = (t1 + t2)
        return termination_kl_loss


class OptimizerGroup:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, state_dicts: Mapping[str, Any], strict: bool = True):
        for optimizer, state_dict in zip(self.optimizers, state_dicts):
            try:
                optimizer.load_state_dict(state_dict, strict=strict)
            except TypeError:
                optimizer.load_state_dict(state_dict)

    def reset(self):
        for optimizer in self.optimizers:
            optimizer.__setstate__({'state': collections.defaultdict(dict)})


class HierarchicalPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,

            option_policy_class, meta_policy_class, initialization_class, termination_class,
            option_policy_kwargs, meta_policy_kwargs, initialization_kwargs, termination_kwargs,
            device, num_options,

            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,

            log_std_init: float = 0.0,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            squash_output: bool = False,
            normalize_images: bool = True,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super(HierarchicalPolicy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            squash_output=squash_output,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            net_arch=[],
            features_extractor_class=UnusedFeaturesExtractor,  # For the moment each module has its own extractor
            features_extractor_kwargs={},
        )
        assert not use_sde, "SDE is not supported for HierarchicalPolicy"
        self.num_options = num_options

        policy_kwargs = setup_hrl(
            option_policy_class, meta_policy_class, initialization_class, termination_class,
            option_policy_kwargs, meta_policy_kwargs, initialization_kwargs, termination_kwargs,
            action_space, device, num_options, observation_space

        )
        option_policies: List[SimpleAC] = policy_kwargs.pop("option_policies")
        meta_policy: StatefulMetaActorCriticPolicy = policy_kwargs.pop("meta_policy")
        initialization: Initiation = policy_kwargs.pop("initialization")
        termination: Termination = policy_kwargs.pop("termination")

        # option_lr_schedule = sb3_utils.get_schedule_fn(option_policy_kwargs.pop("learning_rate"))
        # meta_lr_schedule = sb3_utils.get_schedule_fn(meta_policy_kwargs.pop("learning_rate"))

        self.option_policies = option_policies
        self.meta_policy = meta_policy
        self.initialization = initialization
        self.terminations = termination

        # TODO: this is a hack because the evaluate_policy is hardcoded and not flexible,
        # eval_executing_option_history cuts through several abstractions in a very bug prone way
        # Lives in OptionNet, is filled by an unrelated callback of evaluate_policy, it's consumed by a metric report
        # to be sent to the library manager. This is a mess.

        self.sub_modules = [*self.option_policies, self.meta_policy, self.terminations]
        self.optimizer = OptimizerGroup([sm.optimizer for sm in self.sub_modules])

        param_groups = []
        tracked_params = set()

        for module in self.sub_modules:
            for group in module.optimizer.param_groups:
                group["name"] = module.__class__.__name__
                for param in group["params"]:
                    if param not in tracked_params:
                        param_groups.append(group)
                        tracked_params.add(param)

        del self.mlp_extractor  # this module does not extract features, each sub-module has its own extractor
        del self.action_net  # this module does not predict actions, the options do
        del self.value_net  # this module does not predict values, the meta-policy does

        for n, p in self.named_parameters():
            if p not in tracked_params and p.requires_grad:
                raise ValueError(f"Parameter {n} is not tracked by any optimizer.")

        # device = next(self.parameters()).device
        # self.executing_option = torch.full((num_agents,), constants.NO_OPTIONS, dtype=torch.long, device=device)

    def set_training_mode(self, training_mode):
        self.train(training_mode)
        assert self.option_policies.training == training_mode
        assert self.terminations.training == training_mode
        assert self.meta_policy.training == training_mode

    def forward(self, observation, state, first_transition):
        assert isinstance(state, OptionExecutionState), f"Expected OptionExecutionState, got {type(state)}"
        option_unavailable, unavailable_prob = self.initialization.forward_offpolicy(observation)
        state.option_is_available = torch.logical_not(option_unavailable)
        meta_actions, meta_values, meta_log_probs = self.meta_policy(observation, state.option_is_available)
        meta_values = meta_values.squeeze(1)

        state, termination_probs = self.update_executing_option(observation, state, first_transition, meta_actions)
        assert state.executing_option.max() < self.num_options, f"Executing option is out of bounds"

        termination_probs[first_transition] = 0.0  # The first option can not terminate before it starts
        actions, values, log_probs = (
            torch.full_like(meta_actions, sys.maxsize, dtype=torch.long),
            torch.full_like(meta_values, float("nan")),
            torch.full_like(meta_log_probs, float("nan")),
        )

        for option_idx, option_net in enumerate(self.option_policies):
            option_mask = state.executing_option == option_idx
            if not option_mask.any():
                continue

            option_observation = {k: observation[k][option_mask] for k, v in observation.items()}

            act, val, log_prob = option_net(option_observation)
            actions[option_mask], values[option_mask], log_probs[option_mask] = act, val.squeeze(1), log_prob

        return state, actions, values, log_probs, meta_values, meta_log_probs, termination_probs

    def update_executing_option(self, observation, state: OptionExecutionState, first_transition, meta_actions):
        assert isinstance(observation, dict), f"Expected dict, got {type(observation)}"
        assert isinstance(state, OptionExecutionState), f"Expected OptionExecutionState, got {type(state)}"

        option_terminates, termination_probs = self.terminations(observation, state.executing_option)

        requires_new_option = torch.logical_or(
            torch.logical_or(option_terminates, first_transition),
            torch.eq(state.executing_option, constants.NO_OPTIONS),
        )
        state.executing_option[requires_new_option] = meta_actions[requires_new_option]
        return state, termination_probs

    def predict_values(self, observation, executing_option):
        meta_values = self.meta_policy.predict_values(observation)

        values = torch.full(executing_option.shape, float("nan"), device=executing_option.device)
        for option_idx, policy in enumerate(self.option_policies):
            option_mask = executing_option == option_idx
            if not option_mask.any():
                continue

            if isinstance(observation, dict):
                option_observation = {k: observation[k][option_mask] for k, v in observation.items()}
            else:
                option_observation = observation[option_mask]

            values[option_mask] = policy.predict_values(option_observation).squeeze(1)
        return values, meta_values.squeeze(1)

    def evaluate_actions(self, observation, state, actions):
        meta_values, meta_log_probs, meta_entropies = self.meta_policy.evaluate_actions(observation, state)

        entropies, values, log_probs = (
            torch.full(actions.shape, float("nan"), device=actions.device),
            torch.full(actions.shape, float("nan"), device=actions.device),
            torch.full(actions.shape, float("nan"), device=actions.device),
        )

        for option_idx, policy in enumerate(self.option_policies):
            option_mask, masked_option_observation = get_option_mask(observation, state.executing_option, option_idx)
            if not option_mask.any():
                continue

            distribution = policy.get_distribution(masked_option_observation)

            entropies[option_mask] = distribution.entropy()
            values[option_mask] = policy.predict_values(masked_option_observation).squeeze(1)
            log_probs[option_mask] = distribution.log_prob(actions[option_mask])

        return (meta_values, meta_log_probs, meta_entropies), (values, log_probs, entropies)

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: OptionExecutionState,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        if state is None:
            bs = len(episode_start)

            is_available = torch.zeros((bs, self.num_options), dtype=torch.bool)
            is_available[:, list(self.initialization._available_options)] = True

            state = OptionExecutionState(
                executing_option=torch.full((bs,), constants.NO_OPTIONS, dtype=torch.long),
                option_is_available=is_available,
            )

        old_training_mode = self.training
        self.set_training_mode(False)
        meta_actions = self.meta_policy.predict(observation, state.option_is_available, episode_start, deterministic)
        observation = sb3_utils.obs_as_tensor(observation, self.meta_policy.device)
        meta_actions = torch.tensor(meta_actions)

        state, _ = self.update_executing_option(observation, state, torch.tensor(episode_start), meta_actions)

        actions = torch.empty_like(meta_actions)
        for option_idx, option_net in enumerate(self.option_policies):
            option_mask = state.executing_option == option_idx
            if not option_mask.any():
                continue

            if isinstance(observation, dict):
                option_observation = {k: observation[k][option_mask] for k, v in observation.items()}
            else:
                option_observation = observation[option_mask]

            act, val, log_prob = option_net(option_observation, deterministic=deterministic)
            actions[option_mask] = act

        # Convert to numpy
        actions = actions.cpu().numpy()

        self.set_training_mode(old_training_mode)
        return actions, state


def get_option_mask(observation: torch.Tensor, option: torch.Tensor, option_idx: int) -> (torch.Tensor, torch.Tensor):
    option_mask = option == option_idx
    if isinstance(observation, dict):
        option_observation = {k: observation[k][option_mask] for k, v in observation.items()}
    else:
        option_observation = observation[option_mask]
    return option_mask, option_observation


# class InitializationOC(KLTerminationOptionCritic):
#     def __init__(self, *args, **kwargs):
#         # self.initiation_class: Type[BoostedInitiation] = kwargs.pop("initialization_class")
#         self.initiation_kwargs = kwargs.pop("initialization_kwargs")
#
#         # TODO: this should happen inside the meta policy, wtf
#
#         super().__init__(*args, **kwargs)
#         self.policy.meta_policy_class.initialization = extras.initialization.BoostedInitiation(
#             self.observation_space,
#             **self.initiation_kwargs,
#         ).to(self.device)
#         assert hasattr(self.policy.meta_policy_class,
#                        "initialization"), "The meta policy must have an initialization function"
#
#     def termination_loss(self, locals_, globals_):
#         loss = super().termination_loss(locals_, globals_)
#         loss += self.initialization_loss(locals_, globals_)
#         return loss
#
#     def initialization_loss(self, locals_, _globals):
#         # Q(s, o) > V(s')
#         observations = locals_["observations"]
#         meta_advantages = locals_["meta_advantages"]
#         previous_options = locals_["previous_options"]
#         _, unavailable_prob = self.policy.meta_policy_class.initialization(observations, previous_options)
#         available_prob = 1 - unavailable_prob
#         init_loss = ((self.initiation_margin + meta_advantages) * available_prob).mean()
#
#         self.logger.add_histogram("train/avail_onpolicy", available_prob.detach().numpy().squeeze())
#         self.logger.record("train/avail_onpolicy_mean", available_prob.mean())
#
#         _, unavailable_prob = self.policy.meta_policy_class.initialization.forward_offpolicy(observations)
#         change_points = locals_["previous_options"] != locals_["current_options"]
#         if change_points.any():
#             unavailable_prob[change_points] = 0
#             available_prob = 1 - unavailable_prob
#             self.logger.add_histogram("train/avail_meta", available_prob.detach().numpy().squeeze())
#             self.logger.record("train/avail_meta_mean", available_prob.mean())
#
#         return init_loss


# class InitializationOC(KLTerminationOptionCritic):
#     # def __init__(self, *args, **kwargs):
#     #     self.initiation_class: Type[BoostedInitiation] = kwargs.pop("initialization_class")
#     #     self.initiation_kwargs = kwargs.pop("initialization_kwargs")
#     #     # initialization_class
#
#     #     super().__init__(*args, **kwargs)
#     #     # # TODO: this should happen inside the meta policy, wtf
#     #     # self.policy.meta_policy.initialization = extras.initialization.BoostedInitiation(
#     #     #     self.observation_space,
#     #     #     **self.initiation_kwargs,
#     #     # ).to(self.device)
#     #     # assert hasattr(self.policy.meta_policy,
#     #     #                "initialization"), "The meta policy must have an initialization function"
#
#     def termination_loss(self, locals_, globals_):
#         loss = super().termination_loss(locals_, globals_)
#         loss += self.initialization_loss(locals_, globals_)
#         return loss
#
#     def initialization_loss(self, locals_, _globals):
#         # Q(s, o) > V(s')
#         observations = locals_["observations"]
#         meta_advantages = locals_["meta_advantages"]
#         previous_options = locals_["previous_options"]
#         _, unavailable_prob = self.policy.meta_policy.initialization(observations, previous_options)
#         available_prob = 1 - unavailable_prob
#         init_loss = ((self.initiation_margin + meta_advantages) * available_prob).mean()
#
#         self.logger.add_histogram("train/avail_onpolicy", available_prob.detach().numpy().squeeze())
#         self.logger.record("train/avail_onpolicy_mean", available_prob.mean())
#
#         _, unavailable_prob = self.policy.meta_policy.initialization.forward_offpolicy(observations)
#         change_points = locals_["previous_options"] != locals_["current_options"]
#         if change_points.any():
#             unavailable_prob[change_points] = 0
#             available_prob = 1 - unavailable_prob
#             self.logger.add_histogram("train/avail_meta", available_prob.detach().numpy().squeeze())
#             self.logger.record("train/avail_meta_mean", available_prob.mean())
#
#         return init_loss


class StatefulMetaActorCriticPolicy(MetaAC):
    def evaluate_actions(
            self, obs: torch.Tensor, state: OptionExecutionState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Same as the original policy, but we consider the availability of options.
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # CHANGE VV, it takes the state, because we will want to evaluate the action entropy :(
        distribution = self._get_action_dist_from_latent(latent_pi, state.option_is_available)
        log_prob = distribution.log_prob(state.executing_option)
        # CHANGE ^^

        values = self.value_net(latent_vf)

        return values, log_prob, distribution.entropy()

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            option_is_available: torch.Tensor,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic options.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if episode_start is None:
        #     episode_start = [False for _ in range(self.n_envs)]
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with torch.no_grad():
            option_distribution = self.get_distribution(observation, option_is_available)  # <- CHANGE: we pass the state
            options = option_distribution.get_actions(deterministic=deterministic)

        # Convert to numpy
        options = options.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            raise NotImplementedError("Check the squash_output flag in the original implementation to reimplment")

        # Remove batch dimension if needed
        if not vectorized_env:
            options, = options

        return options

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, option_is_available: torch.Tensor) -> distributions.Distribution:
        option_logits = self.action_net(latent_pi)

        unavail_val = torch.tensor(float("-inf"), device=option_logits.device, dtype=option_logits.dtype)
        masked_option_logits = torch.where(~option_is_available, unavail_val, option_logits)

        dist = self.action_dist.proba_distribution(action_logits=masked_option_logits)
        return dist

    # noinspection PyMethodOverriding
    def forward(self, obs: torch.Tensor,
                option_is_available: torch.Tensor,
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Same as parent except we add the observation in get_action_dist_from_latent so the initialization can be used.
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)

        # CHANGE VV, it takes the state on top of the latent vector
        distribution = self._get_action_dist_from_latent(latent_pi, option_is_available)
        # CHANGE ^^

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def get_distribution(self, obs: torch.Tensor, state: OptionExecutionState) -> distributions.Distribution:
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi, state)  # <-- CHANGE
