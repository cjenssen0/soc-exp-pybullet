# Import packages and environment
import numpy as np
#import gym
# import pybullet_envs as pe
#import pybullet_envs

from spinup.utils.run_utils import ExperimentGrid
from spinup import soc_pytorch
from spinup import sac_pytorch
# from spinup import ppo_pytorch
# from spinup import vpg_pytorch
# from spinup import ddpg_pytorch
import pybullet_envs
import torch as th

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()

    # eg = ExperimentGrid(name='ppo-pyt-bench')
    # eg.add('env_name', 'CartPole-v0', '', True)
    # eg.add('seed', [10*i for i in range(args.num_runs)])
    # eg.add('epochs', 10)
    #eg.add('steps_per_epoch', 1000)
    #eg.add('number_test_episodes', 1)
    # eg.add('ac_kwargs:hidden_sizes', [(32,), (64, 64)], 'hid')
    # eg.add('ac_kwargs:activation', [th.nn.ELU, th.nn.ReLU], '')
    # eg.run(ppo_pytorch)

    eg = ExperimentGrid(name='ppo-hCheetah')
    eg.add('env_name', 'HalfCheetahBulletEnv-v0', '', True)
    # eg.add('seed', [10*i for i in range(args.num_runs)])
    # eg.add('seed', [0])
    eg.add('epochs', 10)
    # eg.add('ac_kwargs:hidden_sizes', [[12, 24, 12]], 'hid')  # [256, 512, 256]
    #eg.add('ac_kwargs:activation', [th.nn.ReLU])
    #eg.add('N_options', [3, 4])
    # eg.add('q_lr', [0.001, 0.002])
    # eg.add('pi_lr', [0.001, 0.002])
    # eg.add('target_kl', [0.01, 0.03, 0.05])
    # eg.add('train_v_iters', [80, 100, 150])
    # eg.add('alpha', [0.1, 0.2])
    # eg.add('c', [0.01, 0.02, 0.03])  # 0.01, 0.02,0.03
    eg.run(sac_pytorch)
