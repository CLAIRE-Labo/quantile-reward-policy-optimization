# Installation with Docker (or any OCI container engine)

## The environment

We provide the following guides for obtaining/building and running the environment:

- To run the image locally (or on a remote server with SSH access) with Docker & Docker Compose, follow the instructions
  to [obtain/build the environment](#obtainingbuilding-the-environment) then
  the instructions [run locally with Docker Compose](#running-locally-with-docker-compose).

  Perform the steps on the machine where the code will run, i.e., your local machine or the remote server.

  The guide also provides instructions to do remote development with VSCode, PyCharm, and Jupyter Lab.
- To run on the CSCS Clariden cluster, follow the instructions in `./CSCS-Clariden-setup/README.md`.

  The guide also provides instructions to do remote development on the Clariden cluster
  with VSCode, PyCharm, and Jupyter Lab.
  Other Slurm + enroot cluster users can get inspiration from it too.

- We also provide an image with the dependencies needed to run the environment
  that you can use with your favorite OCI-compatible container runtime.
  Follow the instructions
  in [Running with your favorite container runtime](#running-with-your-favorite-container-runtime) for the details.
  We give an example of how to run it with Apptainer/Singularity.

## Obtaining/building the environment

### Prerequisites

* `docker` (A Docker Engine `docker version` >= v23). [Install here.](https://docs.docker.com/engine/)
* `docker compose` (`docker compose version` >= v2). [Install here.](https://docs.docker.com/compose/install/)

### Clone the repository

Clone the git repository.

```bash
git clone git@github.com:CLAIRE-Labo/quantile-reward-policy-optimization.git qrpo
cd qrpo
```

### Obtain/build the images

All commands should be run from the `installation/docker-arm64-cuda/` directory.

```bash
cd installation/docker-arm64-cuda
```

1. Create an environment file for your personal configuration with
   ```bash
   ./template.sh env
   ```
   This creates a `.env` file with pre-filled values.
    - The `USRID` and `GRPID` are used to give the container user read/write access to the storage that will be mounted
      when the container is run with a container setup that does not change the user namespace
      (typically the case with rootful Docker and on the EPFL runai cluster).
      Edit them so that they match the user permissions on the mounted volumes, otherwise you can leave them as is.
      (If you're deploying locally, i.e., where you're building, these values should be filled correctly by default.)

    - `LAB_NAME` will be the first element in name of the local images you get.

    - You can ignore the rest of the variables after `## For running locally`.
      These don't influence the build, they will be used later to run your image.

2. Pull or build the generic image.
   This is the image with root as user.
   It will be named according to the image name in your `.env`.
   It will be tagged with `<platform>-root-latest` and if you're building it,
   it will also be tagged with the latest git commit hash `<platform>-root-<sha>` and `<platform>-root-<sha>`.
    - Pull the generic image if it's available.
      ```bash
      # Pull the generic image if available.
      ./template.sh pull_generic skandermoalla/qrpo
      ```
    - Otherwise, build it.
      ```bash
      ./template.sh build_generic
      ```
3. You can run quick checks on the image to check it that it has what you expect it to have:
   ```bash
   # Check all your dependencies are there.
   ./template.sh list_env

    # Get a shell and check manually other things.
    # This will only contain the environment and not the project code.
    # Project code can be debugged on the cluster directly.
    ./template.sh empty_interactive
   ```

4. Build the image configured for your user. (Not needed to run in containers with new namespaces like for the SCITAS and CSCS clusters)
   ```bash
   ./template.sh build_user
   ```
   This will build a user layer on top of the generic image
   and tag it with `*-${USR}` instead of `*-root`.
   This will be the image that you run and deploy to match the permissions on your mounted storage in container
   setups that maintain the user namespace (e.g., rootful Docker).

For the local deployment option with Docker Compose, follow the instructions below.

### Push your image to some Docker registry

Push the generic image if you built it (`LAB_NAME/USR/PROJECT_NAME:PLATFORM-root-latest`).

```bash
./template.sh push_generic docker.io/docker-username/qrpo
```

Pro-tip: it will also push them with the git commit hash as a tag if the build is at the latest commit.
You can rebuild the images with `./template.sh build` to tag them with the latest commit hash.

## Running locally with Docker Compose

**Prerequisites**

Steps prefixed with [CUDA] are only required to use NVIDIA GPUs.

* `docker` (A Docker Engine `docker version` >= v23). [Install here.](https://docs.docker.com/engine/)
* `docker compose` (`docker compose version` >= v2). [Install here.](https://docs.docker.com/compose/install/)
* [CUDA] [Nvidia CUDA Driver](https://www.nvidia.com/download/index.aspx) (Only the driver. No CUDA toolkit, etc.)
* [CUDA] The NVIDIA Container Toolkit. [Install here.](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

**Run**

Edit the `.env` file to specify which hardware acceleration to use with the `ACCELERATION` variable.
Supported values are `cpu` and `cuda`.

Then you can run jobs in independent containers running the runtime image.

These containers start with the entrypoint and then run the command you specified.
By default, they are automatically removed after they exit.
The container has an entrypoint that installs the project, checking that the code directory has correctly been mounted.
The Docker Compose run and dev services are already setup to mount the project code and specify its location
to the entrypoint.

You should not need to override the entrypoint of the container, it performs important setups.
It installs the project from its mounted location when specified to avoid hacky imports,
runs the original entrypoint of your base image if it exists,
and execs your command with PID 1.
Only do so if you need to debug the entrypoint itself or if you have a custom use case.

For the runtime service you can run commands directly in independent containers with
```bash
# template_experiment is an actual script that you can run.
./template.sh run your_command
./template.sh run python --version
./template.sh run python -m qrpo.template_experiment some_arg=some_value
# You can pass environment variables to the container with the `-e VAR=VALUE` flag before your command
./template.sh run -e FOO=10 bash -c 'echo $FOO'
# E.g. open a tmux shell, then run containers there
tmux
./template.sh run your_command
# Detach from the tmux.
```

For the development service you can start it and then exec programs in it, or from a tmux shell
```bash
# To get a shell and run commands in the container
./template.sh dev
# To open a container a let it be (remember to stop it)
./template.sh dev -d
```

You can check how `./template.sh run` is running `docker-compose` commands if you want more control over the containers.
You can pass environment variables that the entrypoint can use to facilitate your development experience.
This is described in the following sections.

You should then return to the root README for the rest of the instructions to run our experiments.

We also recommend that you keep two different copies of the repository, one for the development code
and one for the runtime code so that you don't get containers running experiments reading code that
you're simultaneously developing.
You could for example

```bash
mv qrpo qrpo-tmp
mkdir qrpo
mv qrpo-tmp qrpo/dev
# Make sure to rerun your .env so that the new paths are correct.
git clone git@github.com:CLAIRE-Labo/quantile-reward-policy-optimization.git qrpo/run
# Then you can follow the same steps for .../run to run your experiments.
# Remember to generate and edit the .env file there as well.
```

### Development

We support Pycharm Remote Development (Gateway), VSCode Remote Development, and Jupyter Lab.
For remote development, the template will open an SSH server in a dev container
that you can then use to connect your IDE to and do remote development inside with your usual debugging tools.

#### VS Code and PyCharm Professional Remote Development

```bash
# Start the dev container and an SSH server in it. Change the port if it's already used by another project.
./template.sh dev -d -e SSH_SERVER=1 -e SSH_CONTAINER_PORT=2223

# If the container is on your local machine you're good to go.

# If the container is on a remote machine, you should forward the ports to your local machine.
# Run the following on your local machine.
# ssh -N -L port-on-local-machine:container-ip:container-port <USER@HOST>
ssh -N -L 2223:localhost:2223 <USER@HOST> # or anything specified in your ssh config.
```

Add your forwarded server to your local machine's SSH config file.
```bash
# Add the following to your SSH config file (~/.ssh/config)
# If the container is on your local machine, without port forwarding
# Replace localhost by the address of the container and Port by 22.
Host local2223
	HostName localhost
	User <same-username-as-.env>
	Port 2223
	StrictHostKeyChecking no
	UserKnownHostsFile=/dev/null
	ForwardAgent yes
```

In case of issues, you can check the logs of the container with
```bash
# A container ID is outputted when you start the container.
docker logs -f <container-ID>
```

**Notes**

Directories for storing the IDE configurations, extensions, etc.
are mounted to the container to be persisted across development sessions.
You can find them in the `docker-compose.yaml` file.
Each project (defined by its `PROJECT_ROOT_AT` path) will have its own cache directory to avoid conflicts
between containers.

**VS Code**

The idea is to connect to an SSH remote server (the container) as described [here](https://code.visualstudio.com/docs/remote/ssh).

Install the [Remote Development extension in VSCode](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh).
Then connect to the remote server following the steps [here](https://code.visualstudio.com/docs/remote/ssh#_connect-to-a-remote-host).

Open the directory of your project which is mounted in the same location as in your local machine or remote server.

**PyCharm**

The idea is to run the IDE inside the SSH remote server (the container) as described [here](https://www.jetbrains.com/help/pycharm/remote-development-overview.html)

The first time connecting you will have to install the IDE in the container in a mounted directory
so that it is stored on your machine.
After that, or if you already have the IDE stored from another project,
the template will start the IDE mounted in the container at the container creation,
and you will be able to directly connect to it from the JetBrains Gateway client on your local machine.

The first time,
follow the steps [here](https://www.jetbrains.com/help/pycharm/remote-development-a.html#gateway) to install the IDE,
and install it in `/home/YOUR-USERNAME/.jetbrains-server/dist`, not in its default location
(small "installation options..." link).
For the project directory, it should be mounted in the container at the same location as in your local machine.

In the container, locate the name of the PyCharm IDE installed.
It will be at
```bash
ls ${HOME}/.jetbrains-server/dist
# Outputs something like e632f2156c14a_pycharm-professional-2024.1.4
```
Copy the name of this directory to your `.env` file as the value of the `PYCHARM_IDE_AT` variable
so that the template can start the IDE for you next time.
```bash
PYCHARM_IDE_AT=e632f2156c14a_pycharm-professional-2024.1.4
```

The next time you start the development container, the IDE will start at the container start,
and you will find it in your JetBrains Gateway list of projects.
If it's a new host (e.g. one for a simultaneous container on port 2224), you can add it,
and it will detect that the IDE is already installed.

_Configuration_:

* PyCharm's default terminal is bash. Change it to zsh in the Settings -> Tools -> Terminal.
* When running Run/Debug configurations, set your working directory the project root (`$PROJECT_ROOT_AT`), not the script's directory.
* Your interpreter will be
  * the system Python `/usr/bin/python` with the `from-python` option.
  * the Python in your conda environment with the `from-scratch` option, with the conda binary found at `/opt/conda/condabin/conda`.



#### Jupyter Lab

With Jupyter Lab you should have the server running directly in the container
and then forward the ports to your local machine as follows:

```bash
# Start the jupyter server. Change the port if it's already used by another project.
./template.sh dev -d -e JUPYTER_SERVER=1 -e JUPYTER_PORT=8887
# Outputs a container ID.
# Get its logs to get the token.
docker logs -f <container-ID>
# The last line will be something like
# http://hostname:8887/?token=<TOKEN>
# Wait a bit if the server is not ready yet.
# Ctrl-C to quit the logs.

# If the container is on your local machine open the URL
# http://localhost:8887/?token=<TOKEN> on your local machine.

# If the container is on a remote machine, you should forward the ports to your local machine.
# Run the following on your local machine.
# ssh -N -L port-on-local-machine:container-ip:container-port <USER@HOST>
ssh -N -L 8887:localhost:8887 <USER@HOST> # or anything specified in your ssh config.
# Connect to the server with this URL on your local machine http://localhost:8887/?token=TOKEN
```

## Running with your favorite container runtime

An image with the runtime environment running as root
(but with a configured zshell for users specified at runtime as well)
is available at `https://hub.docker.com/repository/docker/skandermoalla/qrpo/`.

The tag is `arm64-cuda-root-latest` for the latest version and `arm64-cuda-root-SHA` for specific commits
referring to previous builds.
You can use your favorite container runtime to run these images.

They have an entrypoint which installs the project with pip
and expects it to be mounted in the container and its location specified with the
environment variable `PROJECT_ROOT_AT`.
E.g., you can mount it at `/project/qrpo` and specify `PROJECT_ROOT_AT=/project/qrpo`.
The entrypoint can then take any command to run in the container and will run it with PID 1.
(If you don't specify the `PROJECT_ROOT_AT`, the entrypoint will skip the project installation and warn you about it.)
It also expects the working directory to be set to `$PROJECT_ROOT_AT`.
We typically mount the project at the same location as the project root on the host.
You can refer to the `run-local-*` services in the `compose.yaml` file or to the `EPFL-runai-setup/README.md` file
for an idea of how this would work locally with Docker or on a Kubernetes cluster interfaced with Run:ai.

For example, on an HPC system with Apptainer/Singularity you could do
```bash
# After cloning the project, inside the PROJECT_ROOT on your system.
# E.g. apptainer pull docker://registry-1.docker.io/library/ubuntu:latest
apptainer pull docker://registry-1.docker.io/skandermoalla/qrpo:arm64-cuda-root-latest

# Location to mount the project, also used by the entrypoint
export PROJECT_ROOT_AT=/project/qrpo
apptainer run \
    -c \
    -B $(pwd):${PROJECT_ROOT_AT} \
    --cwd ${PROJECT_ROOT_AT} \
    --env PROJECT_ROOT_AT=${PROJECT_ROOT_AT} \
    --env PROJECT_NAME=qrpo \
    --env PACKAGE_NAME=qrpo \
    --env WANDB_API_KEY="" \
    --nv qrpo_arm64-cuda-root-latest.sif
# --env PROJECT_ROOT_AT is used by the entrypoint to install the project
# *.sif is the downloaded image.
# -c to not mount all the home directory to avoid spoiling reproducibility
# --nv to use NVIDIA GPUs
```

Return to the root README for the rest of the instructions to run our experiments.

## Instructions to maintain the environment

The environment is based on an image which already contains system and Python dependencies.
Extra dependencies are managed as follows:

System dependencies are managed by `apt`.
Python dependencies are managed by `pip`.

Complex dependencies that may require a custom installation
should have their instructions performed in the `Dockerfile` directly.

There are two ways to add dependencies to the environment:

1. **Manually edit the loose dependency files.**
   This is used the first time you set up the environment.
   It will also be useful if you run into conflicts and have to restart from scratch.
2. **Add/update dependencies interactively** while running a shell in the container to experiment with which
   dependency is needed.
   This is probably what you'll be doing after building the image for the first time.
   Then you can add these to the

In both cases, after any change, a snapshot of the full environment
specification should be captured in a copy of the dependency file (`requirements-freeze.txt`) that you
can later inspect and reuse to build the image without changing the versions of all the dependencies.
We describe how to do so in the Freeze the Environment section.

### Manual editing (before/while building)

- To add `apt` dependencies, edit the `apt.txt` file.
  Put the dependencies needed to build the environment, e.g., compilers, build tools, etc.
  and dependencies to run the environment, e.g., image processing libraries,
  and the utilities that will help you develop in the container, e.g. `htop`, `vim`, etc.

  If you're not familiar with which dependencies are needed, you can start with the minimal set we
  give, and when you encounter errors during the image build, add the missing dependencies to the stage where the error
  occurred.
- To edit `pip` dependencies, edit the `requirements.txt` file.
- To edit the more complex dependencies, edit the `Dockerfile`.

When manually editing the dependency files,
you do not need to specify the specific version of all the dependencies,
You should just specify the major versions of specific dependencies you need.

### Interactively (while developing)

* To add `apt`  dependencies run `sudo apt install <package>` (in a rootful container),
  and `apt install <package>` (in a container with a mounted namespace where you are the new root user).
* To add `pip` dependencies run `pip install <package>`

### Freeze the environment

After any change to the dependencies, it's good to record snapshot of the full environment specification in a
frozen requirements file (`requirements-freeze.txt`).
This is
to ensure that the environment can be reproduced in other builds
and that the dependencies are tracked at any point in time.

To do so, run the following from a login shell in the container.
The script writes a `requirements-freeze.txt` file with the current environment specification.

The script isn't just a `pip freeze` and the file it generates isn't made to recreate the environment from scratch,
it is tightly coupled to the Dockerfile and the base image it uses.
In this sense, packages that are already installed in the base image or installed by the Dockerfile
may not be listed in the file or may be listed without a version
(this is because that may have been installed from wheels not present anymore in the final image).

The purpose of the generated `requirements-freeze.txt` is to be used always at the same stage of the Dockerfile
to install the same set of missing dependencies between its previous stage and its next stage.
(so not reinstall the dependencies already installed in the base image, for example).
In any case,
the Dockerfile also records the snapshots of the dependency files used to generate each stage for debugging that can be
found in the `/opt/template-dependencies/` directory.

```bash
update-env-file
```

The script isn't perfect, and there are some caveats (e.g., packages installed from GitHub with pip),
so have a look at the output file to make sure it does what you want.
The `update-env-file.sh` gives some hints for what to do,
and in any case you can always patch the file manually.

For dependencies that require a custom installation or build, edit the `Dockerfile`.
If one of these complex dependencies shows in the `requirements-freeze.txt` after the freeze,
you have to remove it, so that pip does not pick it up, and it is installed independently in the `Dockerfile`.

For `apt` dependencies add them manually to the `apt-*.txt` files.

## Troubleshooting

### Supporting multiple images

In case you want to support multiple images for the same platform and hardware acceleration, you can do the following.
(Note that this can also be a way
to have both a `from-python` and `from-scratch` image for the same platform and hardware acceleration
if you want to benchmark/switch between them.)

1. Duplicate the installation directory. E.g.,
   ```bash
   mv installation/docker-arm64-cuda installation/docker-arm64-cuda-env1
   cp installation/docker-arm64-cuda-env1 installation/docker-arm64-cuda-env2
   ```
2. Add an additional field to the image tag for each installation directory.
   Edit the `./template.sh` and your `.env` if already created and edit
   ```bash
   IMAGE_PLATFORM=arm64-cuda-envX
   ```
   This is important so that your images are not mixed.
3. Follow the instructions for each installation directory.

### Debugging the Docker build

If your build fails at some point, the build will print the message with the line in the Dockerfile
that caused the error.
Identify the stage at in which the line is: it's the earliest FROM X as Y before the line.
Then add a new stage right before the failing line starting from the stage you identified.
Something like:

```dockerfile
FROM X as Y

RUN something-that-works

# Add this line.
FROM Y as debug

RUN something-that-breaks
```

Then in the `compose.yaml` file, change the `target: runtime-generic` to `target: Y`
(replacing Y with its correct stage name).
Your build will then stop at the line before the failing line.

```bash
# Say you're building the generic images.
./template.sh build_generic
```

You can open a shell in that layer and debug the issue.

```bash
# IMAGE_NAME and IMAGE_PLATFORM can be found in your .env file.
docker run --rm -it --entrypoint /bin/bash ${IMAGE_NAME}:${IMAGE_PLATFORM}-root-latest
```

### My image doesn't build with my initial dependencies.

Try removing the dependencies causing the issue, rebuilding, and then installing them interactively when running the
container.
The error messages will possibly be more informative, and you will be able to dig into the issue.

Alternatively, you can open a container at the layer before the installation of the dependencies,
like described above, and try to install the environment manually.

## Licenses and acknowledgements

This Docker setup is based on the [Cresset template](https://github.com/cresset-template/cresset)
with the LICENSE.cresset file included in this directory.
