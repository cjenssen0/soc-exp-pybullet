#!/bin/bash

####### HOPPPER #########
# SOC
docker run -v $(pwd):/soc soc python /soc/main.py \
--algorithm soc \
--num_epochs 50 \
--alpha 0.2 \
--data_dir /soc/results_local_docker/

# SAC
# docker run -v $(pwd):/soc soc python /soc/main.py \
# --algorithm sac \
# --num_epochs 50 \
# --alpha 0.2 \
# --data_dir /results/

######## Walker ###########


######## KukaRobot ###########