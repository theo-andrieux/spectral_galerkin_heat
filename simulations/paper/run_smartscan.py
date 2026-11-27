"""Paper Figs. 12 and 14: SmartScan case study (Ramani et al., 2022).

Runs the four island-scan strategies and records the thermal uniformity metric
R = std(T_r) / T_m after each island, where T_r is the top-surface temperature
evaluated on a 60 x 60 radial-basis-function (RBF).

Outputs, under <out>/<strategy>/:
  thermal_uniformity.csv          R per island
  island_025.xmf/.h5 ... _100     top-surface temperature

Usage (from the repository root):
  uv run python simulations/paper/run_smartscan.py <config.yaml>
  uv run python simulations/paper/run_smartscan.py <config.yaml> --strategies smartscan
  uv run python simulations/paper/run_smartscan.py <config.yaml> --islands 1
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
import yaml

from spectral_galerkin_heat.core.parameters import SimulationContext
from spectral_galerkin_heat.io_utils.xdmf_io import write_structured_fields
from spectral_galerkin_heat.physics.spectral_ops import reconstruct_surface_temperature
from spectral_galerkin_heat.solvers import build_solver

HERE = Path(__file__).resolve().parent
GCODE_DIR = HERE / "smartscan_gcodes"
STRATEGIES = ("successive", "successive_chessboard", "lhi", "smartscan")

N_ISLANDS = 100
SNAPSHOT_ISLANDS = (25, 50, 75, 100)
T_MELT = 1658.0          # K

# RBF representation: RBF_S x RBF_S centres, epsilon in 1/mm.
RBF_S = 60
RBF_EPS = 0.8

logger = logging.getLogger("smartscan")


class UniformityMetric:
    """Uniformity metric R for an (ny, nx) top-surface field on the plate."""

    def __init__(self, nx, ny, Lx_mm, Ly_mm):
        x_grid = np.linspace(0.0, Lx_mm, nx)
        y_grid = np.linspace(0.0, Ly_mm, ny)
        x_rbf = np.linspace(0.0, Lx_mm, RBF_S)
        y_rbf = np.linspace(0.0, Ly_mm, RBF_S)
        # Separable Gaussian kernels: grid -> centres (M_t) and centres -> centres (M_r).
        self.Mx = np.exp(-(RBF_EPS * (x_grid[:, None] - x_rbf[None, :]))**2)
        self.My = np.exp(-(RBF_EPS * (y_grid[:, None] - y_rbf[None, :]))**2)
        self.Mrx = np.exp(-(RBF_EPS * (x_rbf[:, None] - x_rbf[None, :]))**2)
        self.Mry = np.exp(-(RBF_EPS * (y_rbf[:, None] - y_rbf[None, :]))**2)
        # Normal equations (My^T My) U (Mx^T Mx) = My^T T Mx, solved by eigendecomposition.
        self.dx, self.Vx = np.linalg.eigh(self.Mx.T @ self.Mx)
        self.dy, self.Vy = np.linalg.eigh(self.My.T @ self.My)
        self.DyDx = np.outer(self.dy, self.dx)
        self.reg = 1e-12 * float(np.max(self.DyDx))

    def __call__(self, T_top):
        T = T_top.astype(np.float64)
        rhs = self.My.T @ T @ self.Mx
        W = (self.Vy.T @ rhs @ self.Vx) / (self.DyDx + self.reg)
        U = self.Vy @ W @ self.Vx.T
        T_r = self.Mry @ U @ self.Mrx
        return float(np.std(T_r) / T_MELT)


def _surface(solver):
    T_top = reconstruct_surface_temperature(solver.state.a, solver.state)
    return T_top.get() if hasattr(T_top, "get") else np.asarray(T_top)


def run_strategy(config_path, raw_cfg, strategy, backend, out_root, n_islands):
    cfg = yaml.safe_load(yaml.safe_dump(raw_cfg))    # deep copy
    cfg["laser"]["path"]["file"] = str(GCODE_DIR / f"{strategy}.gcode")
    if backend:
        cfg["simulation"]["backend"] = backend
    context = SimulationContext.from_dict(cfg, config_dir=str(config_path.parent))
    context.backend = cfg["simulation"]["backend"]

    nx, ny, nz = context.geom.n.x, context.geom.n.y, context.geom.n.z
    Lx, Ly, Lz = context.geom.size.x, context.geom.size.y, context.geom.size.z
    dt = float(context.num.dt)
    total_steps = round(context.num.t_end / dt)
    if total_steps % N_ISLANDS:
        raise ValueError(f"duration/dt = {total_steps} steps is not a multiple of {N_ISLANDS} islands")
    steps_per_island = total_steps // N_ISLANDS

    out_dir = out_root / strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    metric = UniformityMetric(nx, ny, Lx * 1e3, Ly * 1e3)

    logger.info("=== %s | backend %s | mesh %dx%dx%d | dt %.6g s | %d steps/island ===",
                strategy, context.backend, nx, ny, nz, dt, steps_per_island)

    solver = build_solver(context)
    solver.initialize(context)

    rows = []
    t = 0.0
    t0 = time.perf_counter()
    iter_sum = 0
    for step in range(1, n_islands * steps_per_island + 1):
        _, metrics = solver.step(t, dt)
        t += dt
        iter_sum += int(metrics.get("n_evap_iter", 0) or 0)
        if step % steps_per_island:
            continue

        island = step // steps_per_island
        T_top = _surface(solver)
        R = metric(T_top)
        T_max = float(metrics.get("T_surface_max", float("nan")))
        mean_iter = iter_sum / steps_per_island
        iter_sum = 0
        elapsed = time.perf_counter() - t0
        logger.info("  island %3d/100  R=%.5f  T_max=%7.1f K  mean_iter=%.2f  elapsed=%.1f s",
                    island, R, T_max, mean_iter, elapsed)
        rows.append({"island": island, "t_s": t, "R": R, "T_max_K": T_max, "mean_picard_iter": mean_iter, "elapsed_s": elapsed})

        if island in SNAPSHOT_ISLANDS:
            write_structured_fields(
                output_base=out_dir / f"island_{island:03d}",
                fields={"temperature": T_top.astype(np.float32).reshape(1, ny, nx)},
                x=np.linspace(0, Lx, nx, dtype=np.float32),
                y=np.linspace(0, Ly, ny, dtype=np.float32),
                z=np.array([Lz], dtype=np.float32),
                time=t,
            )

        with open(out_dir / "thermal_uniformity.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    if rows:
        logger.info("  %s done in %.1f s  mean R=%.5f",
                    strategy, time.perf_counter() - t0, np.mean([r["R"] for r in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="case YAML (fig12_... or fig14_...)")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    parser.add_argument("--backend", choices=["cpu", "gpu"], help="override simulation.backend")
    parser.add_argument("--islands", type=int, default=N_ISLANDS,
                        help="stop after this many islands (default: all 100)")
    parser.add_argument("--out", help="output directory (default: out/<run_tag>)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    out_root = Path(args.out) if args.out else Path("out") / raw_cfg["io"]["run_tag"]
    out_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out_root / "smartscan.log")])
    logger.info("config: %s", config_path)
    for strategy in args.strategies:
        run_strategy(config_path, raw_cfg, strategy, args.backend, out_root, args.islands)


if __name__ == "__main__":
    main()
