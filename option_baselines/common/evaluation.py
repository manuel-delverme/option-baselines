# import collections
import logging
import os
from typing import List, Optional, Tuple, Union

import gym
import numpy as np
# import torch
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import EvalCallback, EventCallback
# from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env import sync_envs_normalization

from option_baselines.aoc.specs import OptionExecutionState


def every_step_callback_evaluate_policy(
        model: "base_class.BaseAlgorithm",
        env: Union[gym.Env, VecEnv],
        callback: Optional[EventCallback],
        n_eval_episodes: int,
        deterministic: bool,
        render: bool = False,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    This is basically a copy of the original evaluate_policy function from stable_baselines3.common.evaluation
    with the following changes:
    - The callback function is called after each step, not after each episode

    This is necessary to track per-step metrics that are not available at the end of an episode.
    """

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    num_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(num_envs, dtype=int)
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // num_envs for i in range(num_envs)], dtype=int)

    current_rewards = np.zeros(num_envs)
    current_lengths = np.zeros(num_envs, dtype=int)
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    if callback:
        callback.on_rollout_start()

    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations, state=states, episode_start=episode_starts, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1

        if callback:
            callback.update_locals(locals())
            callback.on_step()

        for i in range(num_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if dones[i]:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if callback:
        callback.update_locals(locals())
        callback.on_rollout_end()

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


class EvalMetricsCallback(EventCallback):
    def _on_rollout_start(self):
        self.executed_options = []
        # self.option_advantage_gap = collections.defaultdict(list)
        # self.last_executing_options = None

    def _on_step(self) -> bool:
        state: OptionExecutionState = self.locals["states"]
        # policy: ActorCriticPolicy = self.locals["model"].policy
        self.executed_options.append(state.executing_option)

        # if self.last_executing_options is None:
        #     self.last_executing_options = state.executing_option
        #     obs = self.locals["observations"]
        #     obs, _ = policy.obs_to_tensor(obs)
        #     with torch.no_grad():
        #         values, meta_values = policy.predict_values(obs, state.executing_option)
        #     self.option_entry_value = meta_values

        # has_changed_option = torch.not_equal(self.last_executing_options, state.executing_option)
        # current_advantage = meta_values - self.option_entry_value
        # for has_changed, option_idx, option_adv in zip(has_changed_option, self.last_executing_options, current_advantage):
        #     if has_changed:
        #         self.option_advantage_gap[option_idx].append(option_adv)
        return True

    def _on_rollout_end(self):
        executed_options = np.concatenate(self.executed_options)
        self.executed_options.clear()
        option_frequencies, _ = np.histogram(executed_options, bins=list(self.model.available_options), density=True)
        prefix = "deterministic_eval" if self.locals["deterministic"] else "stochastic_eval"
        self.logger.log({f"{prefix}/frequencies": option_frequencies}, commit=False)


class OptionEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs, ):
        self.callback_eval_metrics: EventCallback = kwargs.pop("metrics_callback")

        super(OptionEvalCallback, self).__init__(*args, **kwargs)
        assert self.deterministic is None, "We are going to use both deterministic and non-deterministic evaluation"

    def _on_training_start(self) -> None:
        self.callback_eval_metrics.init_callback(self.model)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        continue_training = True
        # Sync training and eval env if there is VecNormalize
        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                    "and warning above."
                )

        # Reset success rate buffer
        self._is_success_buffer = []

        deterministic_episode_rewards, deterministic_episode_lengths = every_step_callback_evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=True,
            return_episode_rewards=True,
            warn=self.warn,
            # The callback reports to the library as a side effect, we want to report stochastic metrics because they are closer to training settings
            callback=None,  # self.callback_eval_metrics,
        )

        stochastic_episode_rewards, stochastic_episode_lengths = every_step_callback_evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=False,
            return_episode_rewards=True,
            warn=self.warn,
            callback=self.callback_eval_metrics,
        )

        if self.log_path is not None:
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(deterministic_episode_rewards)
            self.evaluations_length.append(deterministic_episode_lengths)

        deterministic_mean_reward, std_reward = np.mean(deterministic_episode_rewards), np.std(deterministic_episode_rewards)
        deterministic_mean_ep_length, std_ep_length = np.mean(deterministic_episode_lengths), np.std(deterministic_episode_lengths)

        stochastic_mean_reward = np.mean(stochastic_episode_rewards)
        stochastic_mean_ep_length = np.mean(stochastic_episode_lengths)

        if self.verbose > 0:
            logging.info(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={deterministic_mean_reward:.2f} +/- {std_reward:.2f}")
            logging.info(f"Episode length: {deterministic_mean_ep_length:.2f} +/- {std_ep_length:.2f}")

        option_eval_metrics = {
            "deterministic_eval/mean_reward": float(deterministic_mean_reward),
            "deterministic_eval/mean_ep_length": float(deterministic_mean_ep_length),
            "stochastic_eval/mean_reward": float(stochastic_mean_reward),
            "stochastic_eval/mean_ep_length": float(stochastic_mean_ep_length),
        }

        mean_reward = max(deterministic_mean_reward, stochastic_mean_reward)
        self.last_mean_reward = mean_reward
        if len(self._is_success_buffer) > 0:
            success_rate = np.mean(self._is_success_buffer)
            if self.verbose > 0:
                print(f"Success rate: {100 * success_rate:.2f}%")
            option_eval_metrics["eval/success_rate"] = float(success_rate)

        option_eval_metrics["time/total_timesteps"] = self.num_timesteps

        if mean_reward > self.best_mean_reward:
            if self.verbose > 0:
                print("New best mean reward!")
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            self.best_mean_reward = mean_reward
            # Trigger callback on new best model, if needed
            if self.callback_on_new_best is not None:
                continue_training = self.callback_on_new_best.on_step()

        option_eval_metrics["eval/best_mean_reward"] = self.best_mean_reward

        # Trigger callback after every evaluation, if needed
        if self.callback is not None:
            continue_training = continue_training and self._on_event()

        self.logger.log(option_eval_metrics, commit=False)
        return continue_training

# def count_options(model: PPOOC, executed_options):
#     option_counts = collections.Counter()
#     for option_idx in available_options:
#         if option_idx not in model.option_idx_to_obj:
#             # This option is being learned, it is not in the library
#             continue
#         option = model.option_idx_to_obj[option_idx]
#         option_counts[option] = (executed_options == option_idx).sum()
#     total = sum(option_counts.values())
#     option_frequencies = {k: v / total for k, v in option_counts.items()}
#     return option_frequencies
