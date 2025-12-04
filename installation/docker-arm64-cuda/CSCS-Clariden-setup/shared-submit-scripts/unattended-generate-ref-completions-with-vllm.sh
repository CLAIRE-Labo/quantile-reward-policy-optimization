#!/bin/bash

#SBATCH -J qrpo-run-generate
#SBATCH -t 4:00:00
#SBATCH -A a-a10
#SBATCH --output=sunattended.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4

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
$WANDB_API_KEY_FILE_AT \
  --container-workdir=$PROJECT_ROOT_AT \
  --container-env=PROJECT_NAME,PACKAGE_NAME \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  bash -c "\
    sleep 60; \
    exec python -m qrpo.generation.generate_ref_completions_with_vllm \
    subpartition_number=\$SLURM_PROCID \
    $*"

# additional options for pyxis
# --container-env to override environment variables defined in the container

exit 0
