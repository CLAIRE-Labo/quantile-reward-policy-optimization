#!/bin/bash

#SBATCH -J qrpo-run-dist
#SBATCH -t 4:00:00
#SBATCH -A a-a10
#SBATCH --output=sunattended-distributed.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1

# Variables used by the entrypoint script
# Change this to the path of your project (can be the /dev or /run copy)
export PROJECT_ROOT_AT=$HOME/projects/qrpo/run
source $PROJECT_ROOT_AT/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/env-vars.sh
export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

srun \
  --container-image=$CONTAINER_IMAGES/$(id -gn)+$(id -un)+qrpo+arm64-cuda-root-latest.sqsh \
  --environment="${PROJECT_ROOT_AT}/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/edf.toml" \
  --container-mounts=\
$PROJECT_ROOT_AT,\
$SCRATCH,\
/iopsstor/scratch/cscs/smoalla/,\
$WANDB_API_KEY_FILE_AT,\
$OPENAI_API_KEY_AT \
  --container-workdir=$PROJECT_ROOT_AT \
  --container-env=PROJECT_NAME,PACKAGE_NAME \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  bash -c "\
  exec accelerate launch \
  --config-file src/qrpo/configs/accelerate/ds-zero1.yaml \
  --num_machines $SLURM_NNODES \
  --num_processes $((4*$SLURM_NNODES)) \
  --main_process_ip $(hostname) \
  --machine_rank \$SLURM_NODEID \
  $*"

# additional options for pyxis
# --container-env to override environment variables defined in the container

exit 0
