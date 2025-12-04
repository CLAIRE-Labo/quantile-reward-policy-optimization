import io
import re
import subprocess
import sys
from pathlib import Path

RED = "\033[0;33m"
BLUE = "\033[0;34m"
GREEN = "\033[0;32m"
PURPLE = "\033[0;35m"
YELLOW = "\033[0;36m"
NC = "\033[0m"

errors_to_match = [
    "Error:",
    "Timeout at NCCL work",
    "DUE TO TIME LIMIT",
    "SIGTERM",
    "srun: error",
    "Segmentation fault",
    "out of memory",
]


def find_first_line(content, substring):
    """Return the first line from content that contains the given substring."""
    # print(substring)
    for line in content.splitlines():
        if substring in line:
            return line
    return None


def find_n_times(content, substring, n=1):
    """Return the first line from content that contains the given substring."""
    count = 0
    for line in content.splitlines():
        if substring in line:
            count += 1
    return count >= n


def find_failed_jobs(
    parent_file,
    success_message,
    find_success_n_times,
    num_resuming_dirs,
    checkpoint_validator,
):
    import argparse

    parser = argparse.ArgumentParser(description="Find failed jobs and their details.")
    parser.add_argument(
        "job_batch_path",
        type=str,
        help="Path to the job batch directory (e.g. 2025-03-10-17-07).",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Only show failed jobs.",
    )
    parser.add_argument(
        "--only-running",
        action="store_true",
        help="Only show running jobs.",
    )
    parser.add_argument(
        "--only-not-started",
        action="store_true",
        help="Only show jobs that have not started yet.",
    )
    parser.add_argument(
        "--sync-wandb",
        action="store_true",
        help="Sync wandb runs.",
    )
    parser.add_argument(
        "--force-sync-wandb",
        action="store_true",
        help="Sync wandb runs.",
    )
    parser.add_argument(
        "--print-all-commands",
        action="store_true",
        help="Print all commands for jobs that didn't start yet.",
    )
    parser.add_argument(
        "--error-to-exclude",
        type=str,
        default="",
        help="Error to exclude from the errors reported output.",
    )

    args = parser.parse_args()

    # Check for conflicting arguments
    if args.only_failed and args.only_running:
        print(
            "Error: Cannot use both --only-failed and --only-running at the same time."
        )
        sys.exit(1)

    if args.sync_wandb or args.force_sync_wandb:
        import wandb

        wandb_api = wandb.Api()

    batch_path = Path(args.job_batch_path)
    parent_file = Path(parent_file).parent.resolve()
    submit_script = parent_file / batch_path / "submit.sh"

    if not submit_script.is_file():
        raise FileNotFoundError(f"submit.sh not found in {batch_path}")

    lines = submit_script.read_text(encoding="utf-8", errors="replace").splitlines()

    succeeded_count = 0
    failed_count = 0
    running_count = 0
    not_started = 0
    excluded_count = 0
    not_started_commands = []
    errored_commands = []
    running_commands = []

    for line in lines:
        job_command = line.strip()
        if not job_command.startswith("sbatch"):
            continue

        # Use a buffer to collect output
        output_buffer = io.StringIO()

        # Pretty header for command being processed
        output_buffer.write(f"\n\n{'#' * 80}\n")
        output_buffer.write(f"{job_command}\n")
        output_buffer.write(f"{'#' * 80}\n")

        # find the std err and std out files from the command.
        err_file = re.search(r"-e\s+([^ ]+)", job_command)
        out_file = re.search(r"-o\s+([^ ]+)", job_command)

        err_file = Path(err_file.group(1))
        out_file = Path(out_file.group(1))

        # Only take the path after the path of this file starting from "reproducibility-scripts"
        parent_file_from_reproducibility = Path(
            *parent_file.parts[parent_file.parts.index("reproducibility-scripts") :]
        )
        err_file = err_file.relative_to(parent_file_from_reproducibility)
        out_file = out_file.relative_to(parent_file_from_reproducibility)

        if not err_file.exists():
            # Job not started yet.
            output_buffer.write(f"{YELLOW}===== 🚦 Job didn't start yet 🚦 ====={NC}\n")
            not_started += 1
            # Save the command for not started jobs
            not_started_commands.append(job_command)
            if not args.only_running and not args.only_failed:
                print(output_buffer.getvalue())
            continue

        output_buffer.write(f"===== error file =====\n{err_file}\n")
        err_content = err_file.read_text(encoding="utf-8", errors="replace").strip()
        output_buffer.write(f"===== output file =====\n{out_file}\n")
        out_content = out_file.read_text(encoding="utf-8", errors="replace").strip()

        # Look for specific issues by checking for keywords in the file content
        no_error = True
        excluded_error = False
        error_lines = {
            error: find_first_line(err_content, error) for error in errors_to_match
        }
        has_error_to_exclude = len(args.error_to_exclude) > 0 and find_first_line(
            err_content, args.error_to_exclude
        )
        if any(error_lines.values()):
            if has_error_to_exclude:
                no_error = False
                excluded_error = True
            else:
                output_buffer.write(f"===== errors =====\n")
                for error, line in error_lines.items():
                    if line:
                        no_error = False
                        output_buffer.write(f"❌❌❌ '{error}' found:\n")
                        output_buffer.write(f"{line}\n")

        # Extract the resuming directory path from a line containing "resuming_dir: /"
        output_buffer.write(f"===== resuming_dir(s) =====\n")

        resuming_dirs = set()
        for line in out_content.splitlines():
            if "resuming_dir: /" in line:
                parts = line.split()
                if len(parts) >= 2:
                    resuming_dirs.add(parts[1])

        checkpoint_corrupted = False

        if len(resuming_dirs) < num_resuming_dirs:
            output_buffer.write(f"❌❌❌ Missing resuming_dir(s) in the output file.\n")
            output_buffer.write(f"Expecting {num_resuming_dirs} resuming_dir(s).\n")
            output_buffer.write(f"Found {len(resuming_dirs)} resuming_dir(s):\n")
            for resuming_dir in resuming_dirs:
                output_buffer.write(f"{resuming_dir}\n")
        else:
            for resuming_dir in resuming_dirs:
                output_buffer.write("Checkpoints in resuming_dir:\n")
                output_buffer.write(f"{resuming_dir}\n")
                # Find all checkpoints in the resuming directory
                checkpoints = sorted(
                    Path(resuming_dir).glob("checkpoint-*"),
                    key=lambda x: int(re.search(r"\d+", x.name).group()),
                )
                if len(checkpoints) == 0:
                    output_buffer.write(f"{PURPLE}No checkpoints found.{NC}\n")
                else:
                    for checkpoint in checkpoints:
                        # if checkpoint is a symlink, skip it
                        if checkpoint.is_symlink():
                            output_buffer.write(
                                f"{YELLOW}Skipping symlink checkpoint: {checkpoint}{NC}\n"
                            )
                            continue
                        # Verify models weights are there
                        if not (checkpoint / checkpoint_validator).is_file():
                            output_buffer.write(
                                f"❌❌❌{checkpoint_validator} NOT found in {checkpoint.name}\n"
                            )
                            output_buffer.write(f"{checkpoint}\n")
                            checkpoint_corrupted = True
                        else:
                            output_buffer.write(
                                f"{PURPLE}✔ {checkpoint.name} is valid.{NC}\n"
                            )

        # Check jobs is done
        job_done = find_n_times(out_content, success_message, n=find_success_n_times)
        if job_done and not no_error and not excluded_error:
            output_buffer.write("❌❌❌ Job gave success message but had errors ❌❌❌")

        # Determine job status
        job_failed = False
        job_running = False

        if job_done:
            # Check no checkpoint was corrupted
            if not checkpoint_corrupted:
                output_buffer.write(
                    f"{GREEN}===== ✅ Job done, no errors, no corrupt checkpoints. ✅ ====={NC}\n"
                )
                succeeded_count += 1

                # Sync wandb runs if requested
                for resuming_dir in resuming_dirs:
                    if args.sync_wandb or args.force_sync_wandb:
                        # Use wandb API to check id run is already synced and successful
                        wandb_runid = Path(resuming_dir).name
                        do_sync = True
                        try:
                            run = wandb_api.run(f"qrpo/{wandb_runid}")
                            if run.state == "finished":
                                output_buffer.write(
                                    f"{GREEN}Run {wandb_runid} already synced and finished.{NC}\n"
                                )
                                do_sync = False
                            else:
                                output_buffer.write(
                                    f"{YELLOW}Run {wandb_runid} not finished on wandb.{NC}\n"
                                )
                        except wandb.errors.CommError:
                            output_buffer.write(
                                f"{RED}Run {wandb_runid} not found in wandb.{NC}\n"
                            )
                        if do_sync or args.force_sync_wandb:
                            # Sync wandb runs
                            run_dir = Path(resuming_dir) / "wandb" / "latest-run"
                            if run_dir.is_dir():
                                output_buffer.write(
                                    f"{GREEN}Syncing wandb run ...{NC}\n"
                                )
                                wandb_command = f"wandb sync {run_dir}"
                                subprocess.Popen(wandb_command, shell=True)

            else:
                output_buffer.write(
                    f"❌❌❌ Job done, but corrupt checkpoints found. See above.\n"
                )
                job_failed = True
                failed_count += 1
        elif no_error:
            # Job still running probably.
            output_buffer.write(
                f"{YELLOW}===== 🏃🏻‍♂️‍➡️ No finish sign, but no error. Job probably still running... 🏃🏻‍♂️‍➡️ ====={NC}\n"
            )
            job_running = True
            running_commands.append(job_command)
            running_count += 1
        else:
            if excluded_error:
                excluded_count += 1
            job_failed = True
            failed_count += 1

        if job_failed and not excluded_error:
            errored_commands.append(job_command)
            output_buffer.write(
                f"{RED}===== ❌ Job did not finish successfully ❌====={NC}\n"
            )
            output_buffer.write(f"{RED}===== 🔍 To rerun use ====={NC}\n")
            output_buffer.write(f"{BLUE}{job_command}{NC}\n")

        if args.only_failed:
            should_print = job_failed and not excluded_error
        elif args.only_running:
            should_print = job_running
        elif args.only_not_started:
            should_print = False  # Should not arrive here anyway.
        else:
            should_print = True

        if should_print:
            print(output_buffer.getvalue())

    # Print summary
    print(f"{'#' * 80}")
    print(f"{GREEN}===== Summary ====={NC}")
    print(f"{GREEN}Succeeded: {succeeded_count}{NC}")
    print(f"{YELLOW}Running: {running_count}{NC}")
    print(f"{RED}Failed: {failed_count - excluded_count}{NC}")
    print(f"{RED}Excluded: {excluded_count}{NC}")
    print(f"{YELLOW}In queue: {not_started}{NC}")
    print(f"{'#' * 80}")

    # Print all commands for jobs that didn't start yet
    if args.only_not_started and not_started_commands and args.print_all_commands:
        print(f"\n{YELLOW}===== Commands for jobs that didn't start yet ====={NC}")
        for cmd in not_started_commands:
            print(f"{BLUE}{cmd}{NC}")
        print(f"{'#' * 80}")

    # Print all commands for jobs that failed
    if args.only_failed and errored_commands and args.print_all_commands:
        print(f"\n{YELLOW}===== Commands for jobs that failed ====={NC}")
        for cmd in errored_commands:
            print(f"{BLUE}{cmd}{NC}")
        print(f"{'#' * 80}")

    # Print all commands for jobs that are still running
    if args.only_running and running_commands and args.print_all_commands:
        print(f"\n{YELLOW}===== Commands for jobs that are still running ====={NC}")
        for cmd in running_commands:
            print(f"{BLUE}{cmd}{NC}")
        print(f"{'#' * 80}")
