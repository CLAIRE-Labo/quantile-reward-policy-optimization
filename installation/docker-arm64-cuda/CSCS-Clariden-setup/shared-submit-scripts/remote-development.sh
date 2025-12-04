#!/bin/bash

#SBATCH -J qrpo-dev
#SBATCH -t 12:00:00
#SBATCH -A a-a10
#SBATCH --output=sremote-development.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1

# Variables used by the entrypoint script
export PROJECT_ROOT_AT=$HOME/projects/qrpo/dev
source $PROJECT_ROOT_AT/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/env-vars.sh
export HF_TOKEN_AT=$HOME/.hf-token
export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1
export SLURM_ONE_REMOTE_DEV=1
export PRE_COMMIT_HOME=$SCRATCH/pre-commit
export SSH_SERVER=1
export NO_SUDO_NEEDED=1

# When running for the first time,
# mkdir -p $HOME/jetbrains-server
# and can copy the IDE from smoalla
# cp -r /users/smoalla/jetbrains-server/dist/a72a92099e741_pycharm-professional-2024.3.3-aarch64 $HOME/jetbrains-server/dist/
export JETBRAINS_SERVER_AT=$HOME/jetbrains-server
export PYCHARM_IDE_AT=a72a92099e741_pycharm-professional-2024.3.3-aarch64

# or
# mkdir -p $HOME/vscode-server
# export VSCODE_SERVER_AT=$SCRATCH/vscode-server

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

srun \
  --container-image=$CONTAINER_IMAGES/$(id -gn)+$(id -un)+qrpo+arm64-cuda-root-latest.sqsh \
  --environment="${PROJECT_ROOT_AT}/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/edf.toml" \
  --container-mounts=\
$PROJECT_ROOT_AT,\
$SCRATCH,\
/iopsstor/scratch/cscs/smoalla/,\
/capstor/scratch/cscs/$USER/,\
/capstor/scratch/cscs/smoalla/,\
$WANDB_API_KEY_FILE_AT,\
$HF_TOKEN_AT,\
$HOME/.gitconfig,\
$HOME/.bashrc,\
$HOME/.oh-my-bash,\
$HOME/.ssh/authorized_keys,\
$JETBRAINS_SERVER_AT,\
$HOME/.ssh/config \
  --container-workdir=$PROJECT_ROOT_AT \
  --container-env=PROJECT_NAME,PACKAGE_NAME \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  sleep infinity

# additional options
# --container-env to override environment variables defined in the container

# You may also want to mount $HOME/.ssh/config if you bypass the proxy for GitHub.

# Draft.
# Here can connect to the container with
# Get the job id (and node id if multinode)
#
# Connect to the allocation
#   srun --overlap --pty --jobid=JOBID bash
# Inside the job find the container name
#   enroot list -f
# Exec to the container
#   enroot exec <container-pid> zsh
