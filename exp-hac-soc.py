# Import packages and environment
import numpy as np
from spinup.utils.run_utils import ExperimentGrid
# from spinup import soc_pytorch
from spinup import sac_pytorch
# from spinup import ppo_pytorch
import torch as th

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=2)
    args = parser.parse_args()

    eg = ExperimentGrid(name='SAC-HOPPER')
   # eg.add('env_name', 'Walker2DBulletEnv-v0', '', True)
    eg.add('env_name', 'HopperBulletEnv-v0', '', True)

    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 50)
    # eg.add('N_options', [1])  # 2,3
    eg.add('ac_kwargs:hidden_sizes', [[128, 256, 128]], 'hid')
    # eg.add('alpha', [0.1, 0.2])
    eg.add('alpha', [0.2])
    # eg.add('c', [0.3])  # 0.1,0.2,0.3 og evt. 0.0
    eg.run(sac_pytorch)
