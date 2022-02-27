import collections
import PIL.Image
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
        print(f"Parsing options {len(self.option_frames)}, {self.num_timesteps}")
        for k, frames in self.option_frames.items():
            print(f"Option: {k}, {len(frames)} frames")
            video_path = self.eval_env.video_folder + f"/{self.num_timesteps}_option{k}_rollout"
            img = np.stack(frames).mean(axis=0).astype(np.uint8)
            np.clip(img * 4, a_max=255, a_min=None, out=img)
            image_size = 256
            img = cv2.resize(img, (image_size, image_size), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(video_path + ".png", img)
            del img
        self.option_frames = collections.defaultdict(list)
        return True

    def _log_options_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        options = locals_["states"]
        env_idx = locals_["i"]
        env = locals_["env"].env.envs[env_idx]
        current_option = options[env_idx]
        self.option_frames[int(current_option)].append(env.render("rgb_array"))
