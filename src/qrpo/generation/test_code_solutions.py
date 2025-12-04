import json
import logging
import math
import re
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from subprocess import DEVNULL

import datasets
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from qrpo.generation.prepare_code_dataset import pre_code

"""Quickly test if the reference solutions in the LeetCode dataset are valid (not thorough, just a check)."""


@hydra.main(
    version_base=None, config_path="../configs", config_name="prepare-code-dataset"
)
def main(config: DictConfig) -> None:
    """
    Convert the dataset to a preference dataset.
    """
    ds = datasets.load_from_disk(config.dataset_args.dataset_path)

    cpu_limit = config.reward_model_args.cpu_limit
    time_limit = config.reward_model_args.time_limit * cpu_limit
    memory_limit_kb = (
        config.reward_model_args.memory_limit_mb_per_cpu * 1024 * cpu_limit
    )

    def run_cmd(cmd: str, timeout: int, row) -> tuple[str, int | str]:
        """
        Run *cmd* in its own shell, silencing all output.
        Returns (cmd, exit_code) or (cmd, 'timeout').
        """
        try:
            completed = subprocess.run(
                cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL, timeout=timeout
            )
            return cmd, completed.returncode
        except subprocess.TimeoutExpired:
            return cmd, -2

    # test the solutions:
    for split in ["train"]:
        commands = []
        codes = []
        for row in tqdm(ds[split]):
            # extract code between the code fences
            solution = re.search(r"```python(.*?)```", row["response"], re.DOTALL)
            if solution:
                solution = solution.group(1)
            else:
                print(f"no solution found in {row['response']}")
                solution = "exit(255)"
            test_code = f"""
{pre_code}
{solution}

{row['test']}

candidate = {row['entry_point']}
check(candidate)
"""
            shell_code = f"ulimit -v {memory_limit_kb} -t {time_limit}; "
            shell_code += f"python -c {shlex.quote(test_code)}"
            wrapped_program = f"podman run \
                --rm \
                --read-only \
                --network=none \
                docker.io/library/python:3.10-slim \
                bash -c {shlex.quote(shell_code)}"
            commands.append(wrapped_program)

        # run in parallel
        time_limit = config.reward_model_args.time_limit * cpu_limit
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = [pool.submit(run_cmd, c, time_limit, None) for c in commands]
            for fut in tqdm(as_completed(futures)):
                cmd, code = fut.result()
                codes.append(code)

        timeouts = sum(c == -2 for c in codes)
        print(f"Timeouts: {timeouts/len(codes)}")
        format_issues = sum(c == 255 for c in codes)
        print(f"Format issues: {format_issues/len(codes)}")
        errors = sum(c > 0 and c < 255 for c in codes)
        print(f"Errors: {errors/len(codes)}")
        print(f"Success: {(len(codes) - timeouts - errors - format_issues)/len(codes)}")


if __name__ == "__main__":
    main()
