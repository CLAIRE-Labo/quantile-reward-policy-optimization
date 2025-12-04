import json
import re
import shlex
import subprocess
import time

from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf

app = FastAPI()


@app.get("/test")
async def root():
    """This function returns a welcome message."""
    return {"message": "Welcome to the FastAPI application!"}


@app.post("/execute_batch_code/")
async def execute_batch_code(request_body: dict):
    """Run a batch of candidate solutions against their unit‑tests inside isolated
    containers orchestrated through SLURM + Podman and return the pass ratio for
    every candidate.
    """
    # ---------------------------------------------------------------------
    # 1. Unpack request ----------------------------------------------------
    # ---------------------------------------------------------------------
    pre_code = request_body["pre_code"]
    batch_ref_model_completions = request_body["batch_ref_model_completions"]
    batch_entrypoints = request_body["batch_entrypoints"]
    batch_tests = request_body["batch_tests"]
    config = OmegaConf.create(request_body["config"])

    batch_solutions = []
    num_solutions_per_batch = 0
    for ref_completions in batch_ref_model_completions:
        solutions = []
        for ref_completion in ref_completions:
            m = re.search(r"```python(.*?)```", ref_completion, re.DOTALL)
            code_body = m.group(1) if m else "exit(66)"  # always fail if missing
            solutions.append(code_body)
        if num_solutions_per_batch == 0:
            num_solutions_per_batch = len(solutions)
        else:
            if not len(solutions) == num_solutions_per_batch:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch size mismatch: {num_solutions_per_batch} != {len(solutions)}",
                )
        batch_solutions.append(solutions)

    num_tasks = len(batch_solutions) * num_solutions_per_batch
    cpus_per_task = int(config.cpu_limit)
    time_limit = int(config.time_limit)  # seconds per test
    mem_per_task_mb = int(config.memory_limit_mb_per_cpu) * cpus_per_task
    max_tests = int(config.max_tests)

    # ---------------------------------------------------------------------
    # 2. Prepare the Python snippet executed inside every Podman container -
    # ---------------------------------------------------------------------
    solutions_json = json.dumps(batch_solutions)
    tests_json = json.dumps(batch_tests)
    entry_json = json.dumps(batch_entrypoints)
    pre_code += "\n"
    pre_code_json = json.dumps(pre_code)
    mem_limit_bytes = mem_per_task_mb * 1024 * 1024

    # The inner snippet is executed inside each container / SLURM task
    inner_python = f"""
import os, json, signal, resource, builtins, sys, multiprocessing, time
from queue import Empty
multiprocessing.set_start_method("fork", force=True)

# -------------------------------------------------------------------------
# Unpack batch‑level payload ------------------------------------------------
solutions = json.loads({repr(solutions_json)})
tests     = json.loads({repr(tests_json)})
entrypts  = json.loads({repr(entry_json)})
pre_code  = json.loads({repr(pre_code_json)})

idx          = int(os.getenv('SLURM_PROCID', '0'))
batch_idx    = idx // {num_solutions_per_batch}
solution_idx = idx % {num_solutions_per_batch}
sol_code     = pre_code + solutions[batch_idx][solution_idx]
test_code    = tests[batch_idx]
entry_expr   = entrypts[batch_idx]

TIME_LIMIT = {time_limit}  # seconds per assert
MEM_LIMIT  = {mem_limit_bytes}  # bytes per process
MAX_TESTS  = {max_tests}

# -------------------------------------------------------------------------
# Safety shims -------------------------------------------------------------
# -------------------------------------------------------------------------
def _fake_exit(code: int | None = 0):
    raise SystemExit(66)

# -------------------------------------------------------------------------
# Naive assert extractor (faster than AST when test format is stable) ------
# -------------------------------------------------------------------------
def _extract_assert_snippets(src: str, limit: int):
    lines, out, in_body = src.splitlines(), [], False
    for ln in lines:
        if ln.lstrip().startswith("def check"):
            in_body = True
            continue
        if in_body:
            if ln.startswith(" "):
                stripped = ln.lstrip()
                if stripped.startswith("assert"):
                    out.append(stripped)
            else:
                break  # de-indent end of function.
        if len(out) >= limit:
            break
    return out

assert_snippets = _extract_assert_snippets(test_code, MAX_TESTS)
total  = len(assert_snippets)
passed = 0
failed = 0

# -------------------------------------------------------------------------
# Helper to run a single assert in its own constrained process ------------
# -------------------------------------------------------------------------
def _run_single(snippet: str) -> bool:
    if not snippet:
        return False

    q = multiprocessing.SimpleQueue()  # pipe‑based, no background thread

    def _worker(queue: multiprocessing.SimpleQueue):
        # resource limits -----------------------------------------------------
        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT, MEM_LIMIT))
        resource.setrlimit(resource.RLIMIT_CPU, (TIME_LIMIT, TIME_LIMIT))
        # shim exit/quit ------------------------------------------------------
        for name in ("exit", "quit"):
            setattr(builtins, name, _fake_exit)
        sys.exit = _fake_exit
        # run user code -------------------------------------------------------
        env: dict[str, object] = dict()
        exec(sol_code, env)
        env["candidate"] = eval(entry_expr, env)
        try:
            exec(snippet, env)
            queue.put(True)
        except BaseException:
            # This does not capture exceptions from rlimit.
            queue.put(False)

    p = multiprocessing.Process(target=_worker, args=(q,))
    p.start()
    p.join(TIME_LIMIT)
    if p.is_alive():
        p.kill()
        p.join()
    if p.exitcode != 0:
        # This captures exceptions from rlimit.
        return False
    try:
        return q.get()
    except Empty:
        return False

# -------------------------------------------------------------------------
# Run all asserts ---------------------------------------------------------
# -------------------------------------------------------------------------
start = time.time()
for _snippet in assert_snippets:
    if _run_single(_snippet):
        passed += 1
    else:
        failed += 1
end = time.time()
dur = round(end - start, 2)
ratio = passed / total if total else 0.0
print('RESULT_BATCH_IDX_' + str(batch_idx) + '_SOLUTION_IDX_' + str(solution_idx) + ':' + str(ratio), flush=True)
print('DURATION_BATCH_IDX_' + str(batch_idx) + '_SOLUTION_IDX_' + str(solution_idx) + ':' + str(dur), flush=True)
"""

    # ---------------------------------------------------------------------
    # 3. Wrap in Slurm + Podman command ------------------------------------
    # ---------------------------------------------------------------------
    shell_code = f"python -c {shlex.quote(inner_python)}"
    wrapped_program = (
        f"srun --ntasks={num_tasks} "
        f"--cpus-per-task={cpus_per_task} "
        f"--mem={int(mem_per_task_mb * num_tasks * 1.5)}M "
        f"--exclusive "
        f"--kill-on-bad-exit=0 "
        f"podman run --rm --read-only --network=none "
        f"--env SLURM_PROCID docker.io/library/python:3.10-slim "
        f"bash -c {shlex.quote(shell_code)}"
    )

    # ---------------------------------------------------------------------
    # 4. Launch -------------------------------------------------------------
    # ---------------------------------------------------------------------

    try:
        start = time.time()
        result = subprocess.run(
            wrapped_program,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        duration = time.time() - start
        stdout, stderr = result.stdout, result.stderr
        ratios = []
        durations = []
        for _ in range(len(batch_solutions)):
            ratios.append([0.0] * num_solutions_per_batch)
            durations.append([0.0] * num_solutions_per_batch)
        for m in re.finditer(
            r"RESULT_BATCH_IDX_(\d+)_SOLUTION_IDX_(\d+):([0-9]*\.?[0-9]+)", stdout
        ):
            b, s, r = int(m.group(1)), int(m.group(2)), float(m.group(3))
            ratios[b][s] = r
        for m in re.finditer(
            r"DURATION_BATCH_IDX_(\d+)_SOLUTION_IDX_(\d+):([0-9]*\.?[0-9]+)", stdout
        ):
            b, s, d = int(m.group(1)), int(m.group(2)), float(m.group(3))
            durations[b][s] = d
        print("RESULTS:")
        print(ratios)
        print("DURATIONS:")
        print(durations)
        print("DURATION:")
        print(duration)
        return {
            "message": "Batch evaluated successfully",
            "results": ratios,
            "durations": durations,
            "stdout": stdout[:10000],
            "stderr": stderr[:10000],
            "duration_seconds": round(duration, 2),
        }
    except Exception as exc:
        # Stuff like OSError(7, 'Argument list too long')
        if isinstance(exc, OSError):
            raise HTTPException(
                status_code=500,
                detail=f"Error while launching subprocess:\n{exc}",
            )
        # Otherwise
        # Stuff like subprocess.CalledProcessError(127, 'srun', ...)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected failure while launching srun:\nSTDOUT: {exc.stdout[:10000]}\nSTDERR: {exc.stderr[:10000]}\nEXCEPTION: {exc}",
        )
