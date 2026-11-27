"""Grid sweep driver for the paper's runtime figures (Figs. 10 and 11).

Runs every (backend, grid) pair of a ``sweep.yaml`` in its own subprocess,
times the time loop, writes the end-of-run volume field and appends one row to
``<out>/runtime.csv``. With ``reference`` set, the relative L2 error of each
run against that grid is written to ``<out>/errors.csv``: each run's cosine
series is evaluated on the reference nodes and the error is integrated there
with the trapezoidal rule.

Usage (from the repository root):
    uv run python simulations/paper/sweep.py <sweep.yaml>
    uv run python simulations/paper/sweep.py <sweep.yaml> --only 96x48x77
    uv run python simulations/paper/sweep.py <sweep.yaml> --errors-only

sweep.yaml keys:
    config:     base case YAML, relative to sweep.yaml
    grids:      list of [nx, ny, nz]
    backends:   list of "cpu" / "gpu"            (default ["cpu"])
    threads:    CPU threads per run; 1 also pins the process to one core
    reference:  [nx, ny, nz] grid used as the error reference (optional)
    overrides:  {"NXxNYxNZ": {section: body}} config sections replaced for
                that grid only (optional)
    out:        output directory, relative to the repository root
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.fft
import yaml

THREAD_VARS = ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS")


def _grid_key(grid):
    return f"{grid[0]}x{grid[1]}x{grid[2]}"


def _tag(backend, grid):
    return f"{backend}_{_grid_key(grid)}"


def _load_sweep(path):
    path = Path(path).resolve()
    with open(path) as f:
        sweep = yaml.safe_load(f)
    sweep["_dir"] = path.parent
    sweep["config"] = (path.parent / sweep["config"]).resolve()
    sweep["out"] = Path(sweep["out"]).resolve()
    sweep.setdefault("backends", ["cpu"])
    sweep["grids"] = [tuple(int(n) for n in g) for g in sweep["grids"]]
    if sweep.get("reference") is not None:
        sweep["reference"] = tuple(int(n) for n in sweep["reference"])
    return sweep


def _field_path(out, backend, grid):
    fields = out / _tag(backend, grid) / "fields"
    hits = sorted(fields.glob("field_step*.xmf"))
    return hits[-1] if hits else None


# ---------------------------------------------------------------------------
# Child process: run one (backend, grid) case
# ---------------------------------------------------------------------------

def run_single(sweep, backend, grid):
    from spectral_galerkin_heat.core.parameters import SimulationContext
    from spectral_galerkin_heat.io_utils import LocalFSIOManager
    from spectral_galerkin_heat.solvers import build_solver

    with open(sweep["config"]) as f:
        config = yaml.safe_load(f)
    config["simulation"]["backend"] = backend
    if sweep.get("threads") is not None:
        config["simulation"]["threads"] = sweep["threads"]
    config["domain"]["mesh"] = {"value": list(grid), "unit": ""}
    for section, body in (sweep.get("overrides") or {}).get(_grid_key(grid), {}).items():
        config[section] = body
    case_dir = sweep["out"] / _tag(backend, grid)
    config["io"] = {
        "output_root": str(sweep["out"]),
        "run_tag": _tag(backend, grid),
        "output_interval": None,
        "at_end": ["full_volume"],
    }

    context = SimulationContext.from_dict(config, config_dir=str(sweep["config"].parent))
    context.backend = backend
    solver = build_solver(context)
    solver.initialize(context)

    io_manager = LocalFSIOManager()
    io_manager.initialize(context)
    # Fixed directory name, without the timestamp added by initialize().
    shutil.rmtree(io_manager.base_dir, ignore_errors=True)
    io_manager.run_id = _tag(backend, grid)
    io_manager.base_dir = str(case_dir)
    for subdir in io_manager.dirs.values():
        (case_dir / subdir).mkdir(parents=True, exist_ok=True)

    dt = context.num.dt
    n_steps = round(context.num.t_end / dt)
    metrics = {}
    picard_total = picard_unconverged = 0
    if backend == "gpu":
        import cupy as cp
        cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for step in range(n_steps):
        state, metrics = solver.step(step * dt, dt)
        picard_total += int(metrics.get("n_evap_iter", 0) or 0)
        picard_unconverged += not metrics.get("picard_converged", True)
    if backend == "gpu":
        cp.cuda.Device().synchronize()
    runtime = time.perf_counter() - t0

    io_manager.process_end(n_steps * dt, n_steps, state, context.laser_path)
    print("SWEEP_RESULT " + json.dumps({
        "backend": backend, "nx": grid[0], "ny": grid[1], "nz": grid[2],
        "dof": grid[0] * grid[1] * grid[2], "steps": n_steps,
        "runtime_s": round(runtime, 2),
        "picard_iter_total": picard_total,
        "picard_unconverged_steps": int(picard_unconverged),
    }), flush=True)


# ---------------------------------------------------------------------------
# Parent process
# ---------------------------------------------------------------------------

def launch(sweep_path, sweep, backend, grid):
    env = dict(os.environ)
    threads = sweep.get("threads")
    if threads is not None:
        for var in THREAD_VARS:
            env[var] = str(threads)
    cmd = [sys.executable, "-u", __file__, str(sweep_path),
           "--single", f"{grid[0]}x{grid[1]}x{grid[2]}", "--backend", backend]
    if threads == 1 and hasattr(os, "sched_setaffinity"):
        cmd = ["taskset", "-c", "0", *cmd]
    print(f"--- {_tag(backend, grid)}", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (sweep["out"] / f"{_tag(backend, grid)}.log").write_text(proc.stdout + proc.stderr)
    sys.stdout.write(proc.stdout[-2000:])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("SWEEP_RESULT "):
            return json.loads(line[len("SWEEP_RESULT "):])
    return None


def spectral_resample(T, shape):
    """Evaluate a node-centred SG field (nodes ``i*L/n``) on the nodes of a finer grid.

    ``T`` holds the values of a cosine series with ``n`` modes per axis on its
    ``n + 1`` nodes; the series is recovered exactly and evaluated on ``shape``
    nodes per axis.
    """
    coeffs = scipy.fft.idctn(np.asarray(T, dtype=np.float64), type=1)
    modes = tuple(slice(0, n - 1) for n in T.shape)
    padded = np.zeros(shape)
    padded[modes] = coeffs[modes]
    return scipy.fft.dctn(padded, type=1)


def compute_errors(sweep):
    from spectral_galerkin_heat.io_utils.compute_L2_error import compute_L2_structured
    from spectral_galerkin_heat.io_utils.xdmf_io import load_xdmf

    rows = []
    for backend in sweep["backends"]:
        ref_path = _field_path(sweep["out"], backend, sweep["reference"])
        if ref_path is None:
            print(f"no reference field for {backend} {sweep['reference']}, skipping errors")
            continue
        ref = load_xdmf(ref_path)
        for grid in sweep["grids"]:
            if grid == sweep["reference"]:
                continue
            fld_path = _field_path(sweep["out"], backend, grid)
            if fld_path is None:
                continue
            fld = load_xdmf(fld_path)
            for a, b in ((fld.x, ref.x), (fld.y, ref.y), (fld.z, ref.z)):
                if not np.allclose([a[0], a[-1]], [b[0], b[-1]], rtol=1e-6, atol=1e-12):
                    raise ValueError(f"{fld_path}: domain differs from the reference")
            T = spectral_resample(fld.T, ref.T.shape)
            norms = compute_L2_structured(ref.T.astype(np.float64), T, ref.x, ref.y, ref.z)
            rows.append({"backend": backend, "nx": grid[0], "ny": grid[1], "nz": grid[2],
                         "dof": grid[0] * grid[1] * grid[2],
                         "L2_abs": norms["L2_abs"], "L2_rel": norms["L2_rel"],
                         "Linf": norms["Linf"]})
    if rows:
        with open(sweep["out"] / "errors.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {sweep['out'] / 'errors.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", help="path to sweep.yaml")
    parser.add_argument("--only", help="run a single grid, e.g. 96x48x77")
    parser.add_argument("--errors-only", action="store_true",
                        help="skip the runs, only recompute errors.csv")
    parser.add_argument("--single", help=argparse.SUPPRESS)
    parser.add_argument("--backend", help=argparse.SUPPRESS)
    args = parser.parse_args()

    sweep = _load_sweep(args.sweep)

    if args.single:
        run_single(sweep, args.backend, tuple(int(n) for n in args.single.split("x")))
        return

    sweep["out"].mkdir(parents=True, exist_ok=True)
    if not args.errors_only:
        grids = sweep["grids"]
        if args.only:
            grids = [tuple(int(n) for n in args.only.split("x"))]
        csv_path = sweep["out"] / "runtime.csv"
        new = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = None
            for grid in grids:
                for backend in sweep["backends"]:
                    row = launch(args.sweep, sweep, backend, grid)
                    if row is None:
                        print(f"!!! {_tag(backend, grid)} failed", flush=True)
                        continue
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        if new:
                            writer.writeheader()
                    writer.writerow(row)
                    f.flush()
        print(f"wrote {csv_path}")

    if sweep.get("reference") is not None:
        compute_errors(sweep)


if __name__ == "__main__":
    main()
