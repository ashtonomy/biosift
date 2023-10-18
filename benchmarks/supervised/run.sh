#!/bin/bash

###############################################################################
# Torchrun local script
# 
# This script will be called for each process. It will receive command
# line args from the parent script which can be used to coordinate jobs. 
# Command line args expected by this script are as follows:
#    $1 hostname
#    $2 path to environment
#    $3 number of nodes
#    $4 number of gpus
#    $5 target to launch (eg "src/run.py --procs 4")
################################################################################

HOSTNAME=$1
ENV_PATH=$2
NNODES=$3
NGPUS=$4
TARGET=$5

# For future use. Gets master ip address
master_ip=`hostname -I | awk '{print $1}'`

# Causes exit from script on failure
set -e

# This is required to load modules
source /etc/profile.d/modules.sh

# Customize modules according to your needs
module add cuda/10.1.243-gcc/9.5.0
module add nccl/2.11.4-1-gcc/9.5.0-cu11_1-nvK40-nvP-nvV-nvA
module add openmpi/3.1.6-gcc/9.5.0-cu11_1-nvK40-nvP-nvV-nvA-ucx
module add anaconda3/2022.05-gcc/9.5.0

# Navigate to the working directory (exit on failure)
# Assumes script is launched from target dir
cd $PBS_O_WOKRDIR

# Activate conda environment
source activate $ENV_PATH

# This is our launcher. It calls the nccl backend and launches distributed training.
torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NGPUS \
  --rdzv_id=12345 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$HOSTNAME:3000 \
  $TARGET

echo "$HOSTNAME" Finished Tasks