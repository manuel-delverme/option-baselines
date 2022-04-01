import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os

import stable_baselines3
import stable_baselines3.common.callbacks
import stable_baselines3.common.policies
import stable_baselines3.common.preprocessing
import stable_baselines3.common.torch_layers
import stable_baselines3.common.type_aliases
import stable_baselines3.common.vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import functools

import option_baselines
#import option_baselines.aoc as aoc
#import option_baselines.common.buffers
#import option_baselines.common.callbacks
#import option_baselines.common.torch_layers
from option_baselines.tabular.environments import *

parser = argparse.ArgumentParser(description='Temporal Finite Automata Args')
parser.add_argument('--debug', type=bool, default=False, metavar='N',
                    help='Run in debug mode (default: False)')
                    
# METRICS
parser.add_argument('--discount', type=float, default=0.99, metavar='G',
                    help='Discount (gamma) during training (default: 0.99)')
parser.add_argument('--training_steps', type=int, default=100000, metavar='N',
                    help='Maximum number of steps used in training (default: 100000)')
parser.add_argument('--log_iterate_every', type=int, default=100, metavar='N',
                    help='How often to log the performance (default: 100)')
parser.add_argument('--slow_log_iterate_every', type=int, default=1000, metavar='N',
                    help='How often to log the performance (slowly) (default: 10*log_iterate every)')
parser.add_argument('--checkpoint_every', type=int, default=100000, metavar='N',
                    help='How often to checkpoint the performance (default: 100000)')
parser.add_argument('--video_every', type=int, default=10000, metavar='N',
                    help='Interval of creating videos (default: 10000)')
                    
                    
eps_decay = 100_000


optimistic_init = True
features = 256
task = "multitask"  # task = "door-key"
conv_channels = 32

# OPTION CRITIC PARAMETERS
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='Default seed (default: 1)')
parser.add_argument('--entropy_regularization', type=float, default=0.01, metavar='G',
                    help='Coefficient of the entropy regulariser (default: 0.01)')
parser.add_argument('--switching_margin', type=float, default=0.01, metavar='G',
                    help='Coefficient of the entropy regulariser (default: 0.01)')
parser.add_argument('--termination_regularization', type=float, default=0.00, metavar='G',
                    help='Coefficient of the termination function regulariser (default: 0.00)')
parser.add_argument('--lr_pi', type=float, default=7e-4, metavar='G',
                    help='Learning rate for the agent policy (default: 7e-4)')
parser.add_argument('--lr_mu', type=float, default=0.1, metavar='G',
                    help='Learning rate for the agent policy (default: 7e-4)')

# OPTION PARAMS
parser.add_argument('--num_options', type=int, default=2, metavar='N',
                    help='Number of Options (default: 2)')
parser.add_argument('--num_envs_per_task', type=int, default=2, metavar='N',
                    help='Number of Environments per task (default: 2)')
parser.add_argument('--num_tasks', type=int, default=1, metavar='N',
                    help='Number of Tasks (default: 1)')

args = parser.parse_args()
num_envs = args.num_envs_per_task * args.num_tasks

def wrap_envs(num_tasks, num_envs_per_task):
    indices = []
    for idx in range(num_tasks):
        indices.extend([idx] * num_envs_per_task)
    envs = stable_baselines3.common.vec_env.DummyVecEnv(env_fns=[functools.partial(emdp_fourrooms.make_env, idx) for idx in indices], )
    envs = stable_baselines3.common.vec_env.VecVideoRecorder(
        envs,
        video_folder="videos/",
        record_video_trigger=should_record_video,
        video_length=args.running_performance_window)
    envs.seed(args.seed)
    return envs

def should_record_video(step):
    global last_recording
    if step - last_recording > args.video_every:
        last_recording = step
        return True
    return False

def main():
    envs = wrap_envs(args.num_tasks, args.num_envs_per_task)
    agent = option_baselines.aoc.AOC(
        meta_policy=MetaActorCriticPolicy,
        policy=PolicyHideTask,
        env=envs,
        num_options=args.num_options,
        ent_coef=args.entropy_regularization,
        term_coef=args.termination_regularization,
        switching_margin=args.switching_margin,
        gamma=args.discount,
    )

    cb = stable_baselines3.common.callbacks.CallbackList([
        option_baselines.common.callbacks.OptionRollout(envs, eval_freq=args.video_every, n_eval_episodes=num_envs),
        wandb.integration.sb3.WandbCallback(gradient_save_freq=100),
        metrics.CallBack(),
    ])
    agent.learn(args.training_steps, callback=cb, log_interval=(args.log_iterate_every // (agent.n_steps * agent.n_envs)) + 1)


last_recording = 0

if __name__ == "__main__":
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main()

