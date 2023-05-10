import collections
from typing import Any, Dict, Optional, Type, Union, Tuple, List, Mapping

import gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common import utils as sb3_utils
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

import option_baselines.aoc
import option_baselines.aoc.policies
from option_baselines.common import buffers
from option_baselines.common import constants


class PPOO(OnPolicyAlgorithm):
    rollout_buffer: Union[buffers.OptionRolloutBuffer, buffers.DictOptionRolloutBuffer]

    def __init__(
            self,
            meta_policy: Type[ActorCriticPolicy],
            policy: Union[str, Type[ActorCriticPolicy]],
            termination_class: Type["option_baselines.aoc.policies.Termination"],

            env: Union[GymEnv, str],
            num_options: int,
            batch_size: int,
            n_epochs: int,
            clip_range: float,
            initial_ent_coef: float,
            final_ent_coef: float,
            learning_rate: None = None,
            n_steps: int = 5,
            gamma: float = 0.99,
            gae_lambda: float = 1.0,
            meta_ent_coef: float = 0.0,
            term_coef=0.01,
            vf_coef: float = 0.5,
            switching_margin: float = 0.0,
            max_grad_norm: float = 0.5,
            rms_prop_eps: float = 1e-5,
            use_rms_prop: bool = True,
            sde_sample_freq: int = -1,
            normalize_advantage: bool = False,
            offpolicy_learning: bool = True,
            tensorboard_log: Optional[str] = None,

            meta_policy_kwargs: Optional[Dict[str, Any]] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            termination_kwargs: Optional[Dict[str, Any]] = None,

            optimizer_kwargs: Optional[Dict[str, Any]] = None,

            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[torch.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        """
        This class wraps meta-policy and option-policies into a single policy-like object.
        """
        if learning_rate is not None:
            raise ValueError("Learning rate is not supported. Use optimizer_kwargs for options and meta-policy.")

        super(PPOO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=initial_ent_coef,
            vf_coef=vf_coef,
            use_sde=False,
            max_grad_norm=max_grad_norm,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                gym.spaces.Box,
                gym.spaces.Discrete,
                gym.spaces.MultiDiscrete,
                gym.spaces.MultiBinary,
            ),
        )
        self.meta_policy_class = meta_policy
        self.termination_class: Type[option_baselines.aoc.policies.Termination] = termination_class

        self.optimizer_kwargs = optimizer_kwargs
        self.meta_policy_kwargs = meta_policy_kwargs
        self.termination_kwargs = termination_kwargs

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range

        self.term_coef = term_coef
        self.meta_ent_coef = meta_ent_coef
        self.switching_margin = switching_margin
        self.normalize_advantage = normalize_advantage
        self.offpolicy_learning = offpolicy_learning
        self.num_options = num_options
        self._last_options = torch.full(size=(env.num_envs,), fill_value=constants.NO_OPTIONS)
        self.lr_schedule = Exception("You are looking for either meta_policy_lr_schedule or option_lr_schedule")

        self.entropy_scheduler = sb3_utils.get_linear_fn(
            start=initial_ent_coef, end=initial_ent_coef / final_ent_coef, end_fraction=0.5)

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = torch.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        self.buffer_cls = (
            buffers.DictOptionRolloutBuffer
            if isinstance(self.observation_space, gym.spaces.Dict)
            else buffers.OptionRolloutBuffer
        )

        if _init_setup_model:
            self._setup_model()

    @property
    def _ent_coef(self):
        return self.entropy_scheduler(self._current_progress_remaining)

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: buffers.OptionRolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        dones = self._last_episode_starts

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = sb3_utils.obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs, options, meta_values, option_log_probs, termination_probs = self.policy(
                    obs_tensor, dones)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

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
                current_option=options,
                previous_option=self._last_options,
                meta_value=meta_values,
                option_log_prob=option_log_probs,
            )
            del values

            self._last_obs = new_obs
            self._last_options = options.clone()
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last time-step
            new_obs = sb3_utils.obs_as_tensor(new_obs, self.device)
            values, meta_value = self.policy.predict_values(new_obs, options)

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

        self.n_batches = (self.n_epochs * self.rollout_buffer.actions.size) // self.batch_size

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
                current_options = rollout_data.current_options

                observations = rollout_data.observations

                (meta_values, meta_log_prob, meta_entropy), (
                    action_values, action_log_prob, entropy,) = self.policy.evaluate_actions(
                    observations, current_options, actions)
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

        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(action_value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))

        # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("entropy/entropy_coeff", self._ent_coef, self.num_timesteps)
        self.logger.record("entropy/option", -entropies.mean().item())
        self.logger.record("entropy/meta", -meta_entropies.mean().item())

        self.logger.record("train/grad_norm", grad_norm)
        for k, v in grad_means.items():
            self.logger.record("grad_mean/" + k, v.item())

        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/advantages", advantages.mean().item())
        self.logger.record("meta_train/policy_loss", meta_policy_loss.item())
        self.logger.record("meta_train/value_loss", meta_value_loss.item())
        self.logger.record("meta_train/advantages", meta_advantages.mean().item())

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

    def loss_fn(self, locals_, globals_) -> torch.Tensor:
        loss = 0
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

        margin_loss = ((
                               meta_advantages.detach() + self.switching_margin) * termination_probs).mean()  # TODO: the margin should be scaled by the return
        termination_loss = termination_probs.mean()
        self.logger.record("train/margin_loss", margin_loss.item())
        self.logger.record("train/termination_loss", termination_loss.item())
        self.logger.record("train/termination_mean", termination_probs.mean().item())

        loss = termination_loss * self.term_coef + margin_loss
        return loss

    def auxiliary_loss(self, locals_, globals_):
        return 0.0

    def _setup_lr_schedule(self) -> None:
        option_learning_rate = self.policy_kwargs.pop("learning_rate")
        meta_learning_rate = self.meta_policy_kwargs.pop("learning_rate")

        self.option_lr_schedule = sb3_utils.get_schedule_fn(option_learning_rate)
        self.meta_lr_schedule = sb3_utils.get_schedule_fn(meta_learning_rate)

    def _setup_model(self) -> None:
        """
        Setup:
        # - learning rate schedule
        - random seed
        - rollout buffer
        - option-policies
        - meta-policy
        - termination policy
        - proxy policy (to behave like a normal flat policy)
        """
        self._setup_lr_schedule()
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
        )
        policies = torch.nn.ModuleList()  # Policy is a keyword for stable-baselines.
        for _ in range(self.num_options):
            policies.append(
                self.policy_class(
                    self.observation_space, self.action_space, self.option_lr_schedule, **self.policy_kwargs
                ).to(self.device)
            )

        option_space = gym.spaces.Discrete(self.num_options)
        meta_policy = self.meta_policy_class(
            self.observation_space, option_space, self.meta_lr_schedule, **self.meta_policy_kwargs
        ).to(self.device)

        terminations = self.termination_class(
            self.observation_space,
            num_options=self.num_options,
            lr_schedule=self.option_lr_schedule,
            # Termination should follow the meta policy but it goes to infinity too quickly
            **self.termination_kwargs
        ).to(self.device)
        self.policy = OptionNet(policies, meta_policy, terminations, num_agents=self.n_envs)


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


class OptionNet(torch.nn.Module):
    def __init__(
            self,
            policies: torch.nn.ModuleList,
            meta_policy: torch.nn.Module,
            termination: torch.nn.Module,
            num_agents,
    ):
        super(OptionNet, self).__init__()
        self.policies = policies
        self.meta_policy = meta_policy
        self.terminations = termination

        self.sub_modules = [*self.policies, self.meta_policy, self.terminations]
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

        assert not [p for p in self.parameters() if p not in tracked_params and p.requires_grad]

        self.num_options = len(self.policies)
        device = next(self.parameters()).device
        self.executing_option = torch.full((num_agents,), constants.NO_OPTIONS, dtype=torch.long, device=device)

    def set_training_mode(self, training_mode):
        self.policies.train(training_mode)
        self.terminations.train(training_mode)
        self.meta_policy.train(training_mode)

    def forward(self, observation, first_transition):
        meta_actions, meta_values, meta_log_probs = self.meta_policy(observation)
        meta_values = meta_values.squeeze(1)

        termination_probs = self.update_executing_option(first_transition, meta_actions, observation)

        termination_probs[first_transition] = 0.0  # The first option can not terminate before it starts
        actions, values, log_probs = (
            torch.empty_like(meta_actions),
            torch.full_like(meta_values, float("nan")),
            torch.full_like(meta_log_probs, float("nan")),
        )

        for option_idx, option_net in enumerate(self.policies):
            option_mask = self.executing_option == option_idx
            if not option_mask.any():
                continue

            if isinstance(observation, dict):
                option_observation = {k: observation[k][option_mask] for k, v in observation.items()}
            else:
                option_observation = observation[option_mask]

            act, val, log_prob = option_net(option_observation)
            actions[option_mask], values[option_mask], log_probs[option_mask] = act, val.squeeze(1), log_prob

        return actions, values, log_probs, self.executing_option, meta_values, meta_log_probs, termination_probs

    def update_executing_option(self, first_transition, meta_actions, observation):
        option_terminates, termination_probs = self.terminations(observation, self.executing_option)
        requires_new_option = np.logical_or(option_terminates, first_transition)
        self.executing_option[requires_new_option] = meta_actions[requires_new_option]
        return termination_probs

    def predict_values(self, observation, executing_option):
        meta_values = self.meta_policy.predict_values(observation)

        values = torch.full(executing_option.shape, float("nan"), device=executing_option.device)
        for option_idx, policy in enumerate(self.policies):
            option_mask = executing_option == option_idx
            if not option_mask.any():
                continue

            if isinstance(observation, dict):
                option_observation = {k: observation[k][option_mask] for k, v in observation.items()}
            else:
                option_observation = observation[option_mask]

            values[option_mask] = policy.predict_values(option_observation).squeeze(1)
        return values, meta_values.squeeze(1)

    def evaluate_actions(self, observation, options, actions):
        meta_values, meta_log_probs, meta_entropies = self.meta_policy.evaluate_actions(observation, options)

        entropies, values, log_probs = (
            torch.full(actions.shape, float("nan"), device=actions.device),
            torch.full(actions.shape, float("nan"), device=actions.device),
            torch.full(actions.shape, float("nan"), device=actions.device),
        )

        for option_idx, policy in enumerate(self.policies):
            option_mask, masked_option_observation = get_option_mask(observation, options, option_idx)
            if not option_mask.any():
                continue

            distribution = policy.get_distribution(masked_option_observation)

            entropies[option_mask] = distribution.entropy()
            values[option_mask] = policy.predict_values(masked_option_observation).squeeze(1)
            log_probs[option_mask] = distribution.log_prob(actions[option_mask])

        return (meta_values, meta_log_probs, meta_entropies), (values, log_probs, entropies)

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
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
        self.set_training_mode(False)
        meta_actions, _states = self.meta_policy.predict(observation, state, episode_start, deterministic)
        observation = sb3_utils.obs_as_tensor(observation, self.meta_policy.device)
        meta_actions = torch.tensor(meta_actions)

        executing_option = state

        if executing_option is None:
            executing_option = meta_actions.clone()
        else:
            _ = self.update_executing_option(episode_start, meta_actions, observation)

        actions = torch.empty_like(meta_actions)
        for option_idx, option_net in enumerate(self.policies):
            option_mask = executing_option == option_idx
            if not option_mask.any():
                continue

            if isinstance(observation, dict):
                option_observation = {k: observation[k][option_mask] for k, v in observation.items()}
            else:
                option_observation = observation[option_mask]

            act, val, log_prob = option_net(option_observation)
            actions[option_mask] = act

        # Convert to numpy
        actions = actions.cpu().numpy()
        return actions, executing_option


def get_option_mask(observation: torch.Tensor, option: torch.Tensor, option_idx: int) -> (torch.Tensor, torch.Tensor):
    option_mask = option == option_idx
    if isinstance(observation, dict):
        option_observation = {k: observation[k][option_mask] for k, v in observation.items()}
    else:
        option_observation = observation[option_mask]
    return option_mask, option_observation
