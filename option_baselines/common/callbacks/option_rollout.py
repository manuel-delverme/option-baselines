import collections
import os
from typing import Any, Dict
from typing import Optional, Union

import cv2
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
            deterministic: bool = False,

    ):
        super(OptionRollout, self).__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
        )
        self.option_frames = collections.defaultdict(list)
        self.last_log = self.eval_freq
        os.makedirs(os.path.abspath("videos"), exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq is None:
            return True
        if (self.num_timesteps - self.last_log) <= self.eval_freq:
            return True
        if self.model.policy.meta_policy.initialization.available_options < 2:
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
        print(f"Parsing options: {len(self.option_frames)}, timestep: {self.num_timesteps}")

        for k, frames in self.option_frames.items():
            video_path = self.eval_env.video_folder + f"/{self.num_timesteps}_option{k}_rollout"
            img = 255 - np.stack(frames).mean(axis=0).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            np.clip(img * 4, a_max=255, a_min=None, out=img)
            cv2.imwrite(video_path + ".png", img)
            del img

        self.option_frames = collections.defaultdict(list)
        return True

    def _log_options_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        options = locals_["states"]
        env_idx = locals_["i"]
        env = locals_["env"].envs[env_idx]
        current_option = options[env_idx]
        self.option_frames[int(current_option)].append(env.render("rgb_array"))
