import numpy as np

from spinup.utils.run_utils import ExperimentGrid
from spinup import soc_pytorch
from spinup import sac_pytorch
from spinup import ppo_pytorch
from spinup import vpg_pytorch
from spinup import ddpg_pytorch
from spinup import td3_pytorch

import pybullet_envs
import torch as th

# ENV_STRING = ['HalfCheetahBulletEnv', 'Walker2DBulletEnv', 'HopperBulletEnv', 'HumanoidBulletEnv', 'AntBulletEnv']
ENV_STRING = ['Walker2DBulletEnv', 'HopperBulletEnv']

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--num_options', type=int, nargs='+', default=3)
    parser.add_argument('--alpha', nargs='+', type=float, default=.1)
    parser.add_argument('--c', nargs='+', type=float, default=.01)
    parser.add_argument('--environment', type=int, default=0)
    parser.add_argument('--algorithm', type=str, default='soc')
    parser.add_argument('--hid', type=list, default=[128,256,128])
    parser.add_argument('--data_dir', type=str, default='/storage/soft-option-critic-experiments/')

    args = parser.parse_args()

    environment = ENV_STRING[args.environment]

    exp_name = environment \
            + '_algo_' \
            + str(args.algorithm) \
            + '_num_epochs_' \
            + str(args.num_epochs) \
            + '_num_options' \
            + str(args.num_options) \

    eg = ExperimentGrid(name=exp_name)

    eg.add('env_name', environment + '-v0', '', True)

    eg.add('seed', [10*i for i in range(args.num_runs)])

    eg.add('epochs', args.num_epochs)

    if args.algorithm == 'soc':
        eg.add('N_options', args.num_options)

    # eg.add('start_steps', 0)
    eg.add('alpha', args.alpha)
    eg.add('c', args.c)

    if args.algorithm == 'soc':
        eg.run(soc_pytorch, data_dir=args.data_dir)
    elif args.algorithm == 'sac':
        eg.run(sac_pytorch, data_dir=args.data_dir)
    elif args.algorithm == 'ppo':
        eg.run(ppo_pytorch, data_dir=args.data_dir)
    elif args.algorithm == 'vpg':
        eg.run(vpg_pytorch, data_dir=args.data_dir)
    elif args.algorithm == 'ddpg':
        eg.run(ddpg_pytorch, data_dir=args.data_dir)
    elif args.algorithm == 'td3':
        eg.run(td3_pytorch, data_dir=args.data_dir)
