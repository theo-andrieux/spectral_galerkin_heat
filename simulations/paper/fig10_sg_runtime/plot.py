"""Paper Fig. 10: (a) runtime vs DoF, (b) runtime vs self-convergence error.

Run (from the repository root), after the sweep:
    uv run python simulations/paper/fig10_sg_runtime/plot.py [SWEEP_OUT] [FIG_DIR]

SWEEP_OUT defaults to out/paper_fig10_sg_runtime, FIG_DIR to SWEEP_OUT/figures.
Writes runtime_vs_dof.pdf and runtime_vs_error.pdf.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{siunitx}\usepackage{amsmath}\usepackage{bm}\boldmath",
    "font.family": "serif",
    "axes.unicode_minus": False,
    "font.size": 19,
    "axes.labelsize": 21,
    "legend.fontsize": 19,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

STYLE = {"FE": dict(marker="s", label=r"\textbf{FE}"),
         "SG": dict(marker="^", label=r"\textbf{SG}")}
LINE = dict(color="k", linestyle="-", linewidth=3.0, markersize=11,
            fillstyle="none", markeredgewidth=1.8)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def log_axis(axis):
    axis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    axis.set_major_formatter(FuncFormatter(
        lambda v, _: rf"$10^{{{int(round(np.log10(v)))}}}$" if v > 0 else ""))
    axis.set_minor_formatter(NullFormatter())


def figure(series, xlabel, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    handles = [ax.plot(x, y, marker=STYLE[name]["marker"], label=STYLE[name]["label"], **LINE)[0]
               for name, x, y in series]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"\textbf{Runtime (s)}")
    log_axis(ax.xaxis)
    log_axis(ax.yaxis)
    ax.grid(True, which="major", ls="--", color="lightgray", linewidth=0.7)
    ax.set_ylim(top=ax.get_ylim()[1] * 3)   # room for the legend
    ax.legend(handles=handles, loc="best", ncol=len(handles), frameon=True, edgecolor="black")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    sweep_out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/paper_fig10_sg_runtime")
    fig_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else sweep_out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fe_path = sweep_out / "fe_results.csv"
    fe = read_csv(fe_path) if fe_path.exists() else []
    fe_dof = np.array([float(r["dof"]) for r in fe])
    fe_rt = np.array([float(r["runtime_s"]) for r in fe])
    fe_err = [(float(r["L2_rel"]) * 100, float(r["runtime_s"])) for r in fe if r["L2_rel"]]

    sg_rt = {(r["nx"], r["ny"], r["nz"]): r for r in read_csv(sweep_out / "runtime.csv")}
    sg_dof = np.array([float(r["dof"]) for r in sg_rt.values()])
    sg_time = np.array([float(r["runtime_s"]) for r in sg_rt.values()])
    sg_err = [(float(r["L2_rel"]) * 100, float(sg_rt[(r["nx"], r["ny"], r["nz"])]["runtime_s"]))
              for r in read_csv(sweep_out / "errors.csv")]

    o_fe, o_sg = np.argsort(fe_dof), np.argsort(sg_dof)
    fe_series = [("FE", fe_dof[o_fe], fe_rt[o_fe])] if fe else []
    figure(fe_series + [("SG", sg_dof[o_sg], sg_time[o_sg])],
           r"\textbf{Degrees of Freedom (DoF)}", fig_dir / "runtime_vs_dof.pdf")

    fe_err.sort(key=lambda p: p[1])
    sg_err.sort(key=lambda p: p[1])
    fe_series = [("FE", *map(np.array, zip(*fe_err)))] if fe_err else []
    figure(fe_series + [("SG", *map(np.array, zip(*sg_err)))],
           r"$\bm{L^2}$ \textbf{Relative Error (vs finest) (\%)}", fig_dir / "runtime_vs_error.pdf")


if __name__ == "__main__":
    main()
