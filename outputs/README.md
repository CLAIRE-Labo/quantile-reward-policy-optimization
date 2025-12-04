# Instructions for the outputs (models weights, logs, etc.)

## [TEMPLATE] Where and how to set up the outputs

The template provides the `PROJECT_ROOT/outputs/` directory as a placeholder for the outputs generated in the project
(model weights, logs, etc.).
This allows the experiment code to always refer to the same path for the outputs independently of the deployment method
for better reproducibility between deployment options and platforms.
The directory can be accessed in the experiments with `config.outputs_dir`.
The output directories in `PROJECT_ROOT/outputs/` don't need to be physically in the same directory
as the project, you can create symlinks to them.

The default setup config `src/template_package_name/configs/setup.yaml` defines an outputs subdirectory where it will
save the outputs.
This is by default `PROJECT_ROOT/outputs/dev` (so you can symlink that location to somewhere else).
The scripts in the `reproducibility-scripts` directory are set up to write to `PROJECT_ROOT/outputs/shared`.
