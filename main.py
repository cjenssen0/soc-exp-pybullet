# Import packages and environment
import numpy as np
#import gym
# import pybullet_envs as pe
#import pybullet_envs

from spinup.utils.run_utils import ExperimentGrid
from spinup import soc_pytorch
# from spinup import sac_pytorch
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

    eg = ExperimentGrid(name='kuka2-3w-soc')
    # eg.add('env_name', 'HalfCheetahBulletEnv-v0', '', True)
    # eg.add('env_name', 'Walker2DBulletEnv-v0', '', True)
    eg.add('env_name', 'KukaBulletEnv-v0', '', True)

    # eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 25)
    # eg.add('N_options', 3)
    # eg.add('start_steps', 0)
    # eg.add('alpha', [0.1, 0.2])
    # eg.add('c', [0.01, 0.02, 0.03])  # 0.01, 0.02,0.03
    eg.run(soc_pytorch)
