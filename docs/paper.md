# Reproducing Results from the Paper

The folder `simulations/paper/` contains the scripts and configuration files needed to run the
simulations of the paper:

> T. Andrieux, A. Ntinos, M. V. Upadhyay, *Efficient spectral Galerkin framework for nonlinear
> transient heat transfer in finite domains*.

Run every command from the repository root. The GPU cases need the GPU environment:

```bash
uv sync --extra gpu
```

## Overview

| Figures | Case | Command | Hardware |
|---|---|---|---|
| 3–5 | `fig03-05_linear/` | `main.py`, then `analytical_references.py` | CPU |
| 6–9 | `fig06-09_nonlinear/` | `main.py` | GPU |
| 10 | `fig10_sg_runtime/` | `sweep.py`, then `plot.py` | one CPU core |
| 11 | `fig11_cpu_vs_gpu/` | `sweep.py` | GPU, and CPU on all cores |
| 12 | `fig12_smartscan_linear/` | `run_smartscan.py` | GPU |
| 14 | `fig14_smartscan_nonlinear/` | `run_smartscan.py` | GPU |

Each case folder contains a `config.yaml` in the same format as the {doc}`examples`. Three scripts
run the cases:

- `simulations/main.py` runs a single configuration.
- `simulations/paper/sweep.py` runs one configuration on a set of grids.
- `simulations/paper/run_smartscan.py` runs four scan strategies and records a thermal-uniformity
  metric after each island.

In the single-scan cases (Figs. 3–11), a Gaussian laser beam of 24 W and radius 30 µm scans a 316L
cuboid at 0.15 m/s. The nonlinear cases (Figs. 6–11) share the same material, with
temperature-dependent conductivity, density and heat capacity, latent heat of fusion, evaporation
at the top face and convection at the bottom face ($h_c = 3000$ W/(m²·K)).

:::{note}
Depending on the version of the code, results may differ slightly. Runtimes depend on the machine;
the hardware used for each run is reported in the paper.
:::

(paper-linear)=
## Figs. 3–5: linear single scan

{download}`config.yaml <../simulations/paper/fig03-05_linear/config.yaml>` scans the beam along the
centerline of a 2 × 1 × 0.5 mm³ domain, on a $256 \times 128 \times 512$ grid, with constant
properties and no surface losses. The `cpu_linear` backend applies the laser source alone, without
latent heat, evaporation or Picard iteration.
{download}`analytical_references.py <../simulations/paper/fig03-05_linear/analytical_references.py>`
evaluates the Eagar–Tsai and Rosenthal solutions on the same grid and compares the SG field with
Eagar–Tsai.

```bash
uv run python simulations/main.py simulations/paper/fig03-05_linear/config.yaml
uv run python simulations/paper/fig03-05_linear/analytical_references.py
```

The first command writes the SG field to `out/<timestamp>_paper_fig03-05_linear/fields/`. The
second reads the latest such run (or the file given with `--sg`), writes
`out/paper_fig03-05_references/eagar_tsai.xmf` and `rosenthal.xmf`, and prints the relative
$L^2$ difference between the SG field and Eagar–Tsai.

(paper-nonlinear)=
## Figs. 6–9: nonlinear single scan

{download}`config.yaml <../simulations/paper/fig06-09_nonlinear/config.yaml>` solves the nonlinear
problem in the same 2 × 1 × 0.5 mm³ domain, on a $256 \times 128 \times 768$ regular grid. It
writes the volume field every 1000 steps and at the end of the run (step 6000, $t = 12$ ms).

```bash
uv run python simulations/main.py simulations/paper/fig06-09_nonlinear/config.yaml
```

The paper compares this field with a finite-element (FE) reference. Given a temperature field in
XDMF format, the `compute_L2_error` tool compares the last time step of both files:

```bash
uv run python -m spectral_galerkin_heat.io_utils.compute_L2_error \
    path/to/your_temperature_field.xdmf \
    out/<run_dir>/fields/field_step006000.xmf \
    --attr-a Temperature --attr-b temperature --no-error-output
```

(paper-runtime)=
## Figs. 10 and 11: complexity and performance comparisons

Both figures use a reduced single scan: a 1 × 0.5 × 0.25 mm³ domain, with the laser starting at
$x = L_x/4$ on the centerline and scanning for 1 ms. A `sweep.yaml` lists the grids to run:

| | Fig. 10 | Fig. 11 |
|---|---|---|
| Grids | $8 \times 5 \times 12$ to $384 \times 192 \times 300$ | $16 \times 8 \times 8$ to $1024 \times 512 \times 512$ |
| Precision | double | single |
| Hardware | one CPU core | GPU, and CPU on all cores |

:::{note}
Fig. 11 runs in single precision on both the GPU and the CPU, so that the two are compared on equal
terms. Most consumer and workstation GPUs deliver only a small fraction of their single-precision
throughput in double precision.
:::

To run the two set of simulations:

```bash
uv run python simulations/paper/sweep.py simulations/paper/fig10_sg_runtime/sweep.yaml
uv run python simulations/paper/fig10_sg_runtime/plot.py
uv run python simulations/paper/sweep.py simulations/paper/fig11_cpu_vs_gpu/sweep.yaml
```

`sweep.py` runs each grid and appends its measured runtime to `runtime.csv`. For Fig. 10 it also
writes `errors.csv`, the relative $L^2$ error of each grid against the finest one, and `plot.py`
draws the two panels of the figure into `out/paper_fig10_sg_runtime/figures/`. Results go to
`out/paper_fig10_sg_runtime/` and `out/paper_fig11_cpu_vs_gpu/`.

`runtime.csv` grows with each sweep; delete it before a new one. The FE curve of Fig. 10 is not
computed here; `plot.py` adds it when the sweep folder contains a `fe_results.csv`.

(paper-smartscan)=
## Figs. 12 and 14: SmartScan case study

These cases reproduce the island scan-strategy study of
[Ramani et al. (2022)](https://doi.org/10.1016/j.addma.2022.102643). A 60 × 60 × 0.2 mm³ 316L plate
is scanned over 100 islands of 5 × 5 mm², each with 25 hatch lines at 600 mm/s and 200 W.
Convection is set to $h = 25$ W/(m²·K).

{download}`fig12 config.yaml <../simulations/paper/fig12_smartscan_linear/config.yaml>` solves the
case in a linear configuration, reproducing the linear thermal model of Ramani et al. It uses their
200 µm in-plane grid, with a finer resolution in $z$ ($300 \times 300 \times 4$), and a time step of
1/3000 s.
{download}`fig14 config.yaml <../simulations/paper/fig14_smartscan_nonlinear/config.yaml>` is the
same case with latent heat and evaporation, on a refined $1200 \times 1200 \times 4$ grid with a
time step of 1/12000 s.

`run_smartscan.py` runs the four strategies one after another. After each island it computes the
thermal-uniformity metric $R$ as defined by Ramani et al.: the top-surface temperature is fitted
with radial basis functions, and $R$ is the standard deviation of the fitted temperature divided by
the melting temperature (1658 K).

```bash
uv run python simulations/paper/run_smartscan.py simulations/paper/fig12_smartscan_linear/config.yaml
uv run python simulations/paper/run_smartscan.py simulations/paper/fig14_smartscan_nonlinear/config.yaml
```

- `--strategies smartscan lhi` runs a subset of `successive`, `successive_chessboard`, `lhi`,
  `smartscan`.
- `--islands 5` stops each strategy after five islands.
- `--backend cpu` overrides `simulation.backend`.

Results go to `out/<run_tag>/<strategy>/`, or to the folder given with `--out`:

- `thermal_uniformity.csv`: one row per island, with $R$, the peak surface temperature and the
  mean number of Picard iterations.
- `island_025.xmf` to `island_100.xmf`: the top-surface temperature after 25, 50, 75 and 100
  islands, readable in ParaView.

The log, `out/<run_tag>/smartscan.log`, prints the mean $R$ of each strategy at its end. These
simulations were initially run on a Quadro RTX 5000: the linear case took about 17 min and the
nonlinear case about 16 h.

### Scan paths

The folder `smartscan_gcodes/` contains the G-code files of the four strategies.

- Island 1 is in the bottom-left corner. The numbers grow along each row and change direction on
  every other row, so consecutive islands are always neighbors.
- Odd-numbered islands have scan lines along $x$, even-numbered islands along $y$.
- The laser jumps between islands take no time.

The successive strategy scans islands 1 to 100 in order. The successive chessboard scans the odd
islands, then the even ones. The LHI and SmartScan orders are those listed in the appendix of
Ramani et al.
