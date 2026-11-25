"""Paper Figs. 3-5: Eagar-Tsai and Rosenthal reference fields.

Evaluates both solutions at the end of the run described by ``config.yaml``, on
the nodes of the SG output grid, and writes them to
``out/paper_fig03-05_references/{eagar_tsai,rosenthal}.xmf``. Then prints the
relative L2 difference between the SG end-of-run field and Eagar-Tsai,
||T_SG - T_ET|| / ||T_ET|| summed over the grid nodes. The SG field is the
latest ``main.py`` run of ``config.yaml`` found under ``out/``, or ``--sg``.

Run (from the repository root):
    uv run python simulations/main.py simulations/paper/fig03-05_linear/config.yaml
    uv run python simulations/paper/fig03-05_linear/analytical_references.py
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import yaml

from spectral_galerkin_heat.analytical import eagar_tsai_field, rosenthal_field
from spectral_galerkin_heat.core.parameters import SimulationContext
from spectral_galerkin_heat.io_utils.xdmf_io import XdmfBuilder, load_xdmf

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"

OUT_ROOT = Path("out")
OUT_DIR = OUT_ROOT / "paper_fig03-05_references"


def load_case():
    """Physical parameters, time step and laser track of the SG run."""
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    ctx = SimulationContext.from_dict(cfg, config_dir=str(HERE))
    n_steps = int(ctx.num.n_steps)
    dt, t_total = float(ctx.num.dt), float(ctx.num.t_end)

    start = ctx.laser_path.get_state(0.0, dt)
    after_one_step = ctx.laser_path.get_state(dt, dt)
    v = (float(after_one_step.x) - float(start.x)) / dt

    return {
        "material": {"rho": float(ctx.mat.rho), "k": float(ctx.mat.k),
                     "Cp": float(ctx.mat.Cp), "T0": float(ctx.mat.T0)},
        "A": float(ctx.laser.absorptivity), "P": float(ctx.laser.power),
        "r_b": float(ctx.laser.radius),
        "size": (float(ctx.geom.size.x), float(ctx.geom.size.y), float(ctx.geom.size.z)),
        "mesh": (int(ctx.geom.n.x), int(ctx.geom.n.y), int(ctx.geom.n.z)),
        "x_end": float(start.x) + v * t_total, "y_laser": float(start.y), "v": v,
        "dt": dt, "t_total": t_total, "n_steps": n_steps,
        "run_tag": cfg["io"]["run_tag"],
    }


def find_sg_output(case):
    """End-of-run field of the latest ``main.py`` run of this case."""
    name = f"field_step{case['n_steps']:06d}.xmf"
    hits = sorted(OUT_ROOT.glob(f"*_{case['run_tag']}/fields/{name}"),
                  key=lambda p: p.stat().st_mtime)
    if not hits:
        raise FileNotFoundError(
            f"no {name} under {OUT_ROOT}/*_{case['run_tag']}/fields/; run first:\n"
            f"  uv run python simulations/main.py {os.path.relpath(CONFIG)}")
    return hits[-1]


def load_sg(path, x, y, z, t_total):
    """SG field, checked against the reference grid and end time."""
    sg = load_xdmf(Path(path), "temperature")
    for name, a, b in (("x", sg.x, x), ("y", sg.y, y), ("z", sg.z, z)):
        a = np.asarray(a, dtype=np.float64)
        if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=1e-6 * b[-1]):
            raise ValueError(f"{path}: {name} grid differs from config.yaml")
    if sg.time is not None and not np.isclose(sg.time, t_total, rtol=1e-5):
        raise ValueError(f"{path}: field at t = {sg.time} s, expected {t_total} s")
    return np.asarray(sg.T, dtype=np.float64)


def save(T, x, y, z, basename, t_total):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h5_path = OUT_DIR / f"{basename}.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X", data=x.astype(np.float32))
        f.create_dataset("Y", data=y.astype(np.float32))
        f.create_dataset("Z", data=z.astype(np.float32))
        f.create_dataset("temperature", data=T.astype(np.float32))
    builder = XdmfBuilder(version="2.0")
    builder.add_structured_grid(name=basename, dims=T.shape, h5_ref=h5_path.name,
                                attributes={"temperature": "temperature"}, time=t_total)
    builder.write(str(OUT_DIR / f"{basename}.xmf"))
    print(f"wrote {OUT_DIR / basename}.xmf   T_max = {T.max():.1f} K")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sg", help="SG end-of-run field (.xmf); default: latest run under out/")
    args = parser.parse_args()

    case = load_case()
    sg_path = Path(args.sg) if args.sg else find_sg_output(case)
    Lx, Ly, Lz = case["size"]
    nx, ny, nz = case["mesh"]
    x = np.arange(nx + 1) * (Lx / nx)
    y = np.arange(ny + 1) * (Ly / ny)
    z = np.arange(nz + 1) * (Lz / nz)
    source = {"A": case["A"], "P": case["P"], "x_laser": case["x_end"],
              "y_laser": case["y_laser"], "v": case["v"]}

    T_sg = load_sg(sg_path, x, y, z, case["t_total"])

    T_et = eagar_tsai_field(x, y, z, **case["material"], **source, r_b=case["r_b"],
                            Lx=Lx, Ly=Ly, Lz=Lz,
                            t_total=case["t_total"], dt=case["dt"], clip_source=True)
    save(T_et, x, y, z, "eagar_tsai", case["t_total"])

    T_ro = rosenthal_field(x, y, z, **case["material"], **source, Ly=Ly, Lz=Lz)
    save(T_ro, x, y, z, "rosenthal", case["t_total"])

    err = T_sg - T_et
    L2_rel = np.sqrt(np.sum(err**2) / np.sum(T_et**2))
    print(f"SG field: {sg_path}")
    print(f"SG vs Eagar-Tsai: L2_rel = {L2_rel * 100:.4f} %   Linf = {np.abs(err).max():.2f} K")


if __name__ == "__main__":
    main()
