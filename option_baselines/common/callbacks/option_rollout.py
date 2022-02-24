import collections
from typing import Any, Dict
from typing import Optional, Union

import gym
import numpy as np
import stable_baselines3.common.evaluation
from stable_baselines3.common import callbacks
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class OptionRollout(callbacks.EvalCallback):
    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            callback_on_new_best: Optional[BaseCallback] = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: Optional[str] = None,
            best_model_save_path: Optional[str] = None,
            deterministic: bool = True,

    ):
        super(OptionRollout, self).__init__(eval_env, callback_on_new_best, n_eval_episodes, eval_freq, log_path, best_model_save_path, deterministic)
        self.option_frames = collections.defaultdict(list)
        self.last_log = self.eval_freq

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_log) <= self.eval_freq:
            return True

        self.last_log = self.num_timesteps
        stable_baselines3.common.evaluation.evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=False,
            deterministic=self.deterministic,
            return_episode_rewards=False,
            warn=False,
            callback=self._log_options_callback,
        )
        for k, frames in self.option_frames.items():
            video_path = self.eval_env.video_folder + f"/{self.num_timesteps}_option{k}_rollout"
            from PIL import Image

            img = np.stack(frames).mean(axis=0)
            img = Image.fromarray(img.astype(np.uint8))
            img.save(video_path + ".png")

        return True

    def _log_options_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        options = locals_["states"]
        env_idx = locals_["i"]
        env = locals_["env"].env.envs[env_idx]
        current_option = options[env_idx]
        self.option_frames[int(current_option)].append(env.render("ascii"))
