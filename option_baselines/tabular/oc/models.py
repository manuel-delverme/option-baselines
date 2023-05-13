import distutils.spawn
import functools
import os

import gym
import stable_baselines3
import stable_baselines3.common.callbacks
import stable_baselines3.common.policies
import stable_baselines3.common.preprocessing
import stable_baselines3.common.torch_layers
import stable_baselines3.common.type_aliases
import stable_baselines3.common.vec_env
import torch
import torch.distributions
import torch.nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PATH"] = f"{os.environ['PATH']}{os.pathsep}{os.environ['HOME']}/ffmpeg/ffmpeg-5.0-i686-static/"

assert distutils.spawn.find_executable("ffmpeg")


class MetaCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        super(MetaCombinedExtractor, self).__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if stable_baselines3.common.preprocessing.is_image_space(subspace):
                raise NotImplementedError
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = torch.nn.Flatten()
                total_concat_size += stable_baselines3.common.preprocessing.get_flattened_obs_dim(subspace)

        self.extractors = torch.nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: stable_baselines3.common.type_aliases.TensorDict) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)


class MetaActorCriticPolicy(stable_baselines3.common.policies.MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = MetaCombinedExtractor
        kwargs["net_arch"] = []
        super(MetaActorCriticPolicy, self).__init__(*args, **kwargs)


class PolicyHideTask(stable_baselines3.common.policies.MultiInputActorCriticPolicy):
    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        obs = {
            "image": obs["image"],
            "task": obs["task"] * 0,
        }
        obs = super().extract_features(obs)
        return obs