import numpy as np
import stable_baselines3.common
import torch

import hyper


class CallBack(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(self):
        super(CallBack, self).__init__()
        self.last_log = 0
        task_idx_mask = torch.div(torch.arange(hyper.num_envs), hyper.num_envs_per_task, rounding_mode="trunc")
        self.task_idx_mask = task_idx_mask.unsqueeze(0)

    def _on_step(self):
        if not (self.n_calls % 10) != 0:
            return

        meta_policy = self.locals["self"].policy.meta_policy
        obs_tensor = self.locals["obs_tensor"]
        features = meta_policy.extract_features(obs_tensor)
        latent_pi, latent_vf = meta_policy.mlp_extractor(features)
        task_idx_mask = self.task_idx_mask.squeeze()
        num_tasks = hyper.num_tasks
        distribution = meta_policy._get_action_dist_from_latent(latent_pi)
        for task_idx in range(num_tasks):
            task_dist = distribution.distribution.probs.detach()
            self.logger.add_histogram(f"task{task_idx}/meta_distr", task_dist[task_idx_mask == task_idx], self.num_timesteps)

        super(CallBack, self)._on_step()

    def _on_rollout_end(self):
        super(CallBack, self)._on_rollout_end()

        if (self.num_timesteps - self.last_log) <= hyper.log_iterate_every:
            return
        print("progress", 1 - self.locals["self"]._current_progress_remaining)
        self.last_log = self.num_timesteps
        rollout_steps = self.locals["self"].n_steps
        num_tasks = hyper.num_tasks
        rollout_buffer = self.locals["rollout_buffer"]
        executed_options = rollout_buffer.current_options
        switches = rollout_buffer.current_options != rollout_buffer.previous_options
        switches[rollout_buffer.episode_starts.astype(bool)] = False

        self.logger.add_scalar("rollout/mean_returns", rollout_buffer.returns.mean(), self.num_timesteps)
        self.logger.add_scalar("rollout/mean_rewards", rollout_buffer.rewards.mean(), self.num_timesteps)
        self.logger.add_scalar("rollout/change_points", switches.mean(), self.num_timesteps)
        self.logger.add_scalar("rollout/action_gap", np.inf, self.num_timesteps)

        self.logger.add_histogram("rollout/executed_options", executed_options, self.num_timesteps)

        env_returns = rollout_buffer.returns.mean(0)
        task_returns = env_returns.reshape(num_tasks, -1).mean(1)

        env_rewards = rollout_buffer.rewards.sum(0)
        task_rewards = env_rewards.reshape(num_tasks, -1).mean(1)

        task_idx_mask = self.task_idx_mask.repeat(rollout_steps, 1)
        for task_idx in range(num_tasks):
            task_executed_option = executed_options[task_idx_mask == task_idx]
            if len(task_executed_option) < 3:
                task_executed_option = np.tile(task_executed_option, 3)[:3]
            self.logger.add_histogram(f"task{task_idx}/task_executed_options", task_executed_option, self.num_timesteps)

            task_return = task_returns[task_idx]
            task_reward = task_rewards[0]
            self.logger.add_scalar(f"task{task_idx}/mean_return", task_return, self.num_timesteps)
            self.logger.add_scalar(f"task{task_idx}/mean_rewards", task_reward, self.num_timesteps)