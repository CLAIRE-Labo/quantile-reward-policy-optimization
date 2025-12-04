#!/bin/bash

#SBATCH -J qrpo-run-reward
#SBATCH -t 4:00:00
#SBATCH -A a-a10
#SBATCH --nodes 1

# Variables used by the entrypoint script
# Change this to the path of your project (can be the /dev or /run copy)
export PROJECT_ROOT_AT=$HOME/projects/qrpo/run
source $PROJECT_ROOT_AT/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/env-vars.sh
export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Node
# 288 cpus
# ~460 GB

# CPUS
# 64 CPUs for main program
# 64 CPUs for machine (and sandbox sever)
# 50 CPUs for sandbox programs
# Sandbox
# 1 CPUs per solution. (N tests sequentially)
# 50 solutions simultaneously, 1 ref completions, 50 prompts.

# Memory
# 64 GB for main program
# 64 GB for machine (and sandbox server)
# 75 GB for sandbox programs (1.5GB per cpu. 50 cpus)

# load sandbox execution image
podman load -i $CONTAINER_IMAGES/python3.10-slim.tar

# Start the sandbox server.
mamba run \
  --cwd $PROJECT_ROOT_AT/src/sandbox \
  -n sandbox-server \
  --live-stream \
  uvicorn server:app \
  --host localhost \
  --port 8565 &

sleep 5

# Start the compute script.
srun \
  --exclusive \
  --ntasks=1 \
  --cpus-per-task=64 \
  --mem=64G \
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
  "$@"

# additional options for pyxis
# --container-env to override environment variables defined in the container

exit 0
