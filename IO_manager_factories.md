# Plan — Reviewer Tasks 4 (IOManager) & 5 (Factories)

This plan covers the two outstanding reviewer items. Tasks 1–3 (unify kernel
definitions, `MathBackend` abstraction, unified CPU/GPU solver) are already
done.

## Guiding principles

- **Stay efficient with the code writing.** Reuse what already exists
  (`get_backend`, the unified `SpectralSolver`, `LocalFSIOManager`); change only
  what each task requires. No speculative abstractions, no new layers beyond
  what the reviewer asked to *remove*.
- **Prioritize readability.** The point of both tasks is to delete indirection
  that isn't paying for itself (a one-implementation ABC, three near-identical
  factory classes) and to make `save_step` legible via small, named helpers and
  a `match`/`case` dispatch. Prefer fewer, clearer call sites over clever ones.

---

## Task 5 — Simplify / remove factories

**Problem (reviewer):** `CPUSimulationFactory`, `GPUSimulationFactory`, and
`CPULinearSimulationFactory` all build the *same* `LocalFSIOManager` and differ
only in the solver. The abstract-factory hierarchy buys nothing.

**Approach — replace the factory hierarchy with one selector function + direct
injection into the runner.**

1. Add `build_solver(context) -> HeatSolver` (in `solvers/__init__.py`). It holds
   the `(method, backend)` dispatch currently buried in `simulations/main.py:get_factory`:
   - `spectral` + `cpu`        → `SpectralSolver(NumpyBackend())`
   - `spectral` + `gpu`        → `SpectralSolver(get_backend("cupy"))`
   - `spectral` + `cpu_linear` → `SpectralSolverCPULinear()`
   - `fem` / unknown           → raise (keep the existing extension-point comment)
2. Change `StandaloneHeatRunner.__init__` to take the components directly:
   `__init__(self, context, heat_solver, io_manager=None, config=None, yaml_path=None)`,
   defaulting `io_manager` to `LocalFSIOManager()`. Drop the `factory` param and
   the two `factory.create_*()` calls.
3. Update consumers:
   - `simulations/main.py`: replace `get_factory(context)` with
     `build_solver(context)`; pass it as `heat_solver=`. The GPU-OOM fallback path
     rebuilds the solver the same way.
   - `tests/test_integration.py`: the three call sites
     (`CPUSimulationFactory(context)`, `GPUSimulationFactory(context)`,
     `factory_cls(context)`) become `build_solver(context)` /
     `SpectralSolver(NumpyBackend())` as appropriate.
4. Delete the `factories/` package: `base.py`, `cpu_factory.py`,
   `gpu_factory.py`, `cpu_linear_factory.py`, `__init__.py`.

**Net effect:** ~5 files deleted, one small free function added, runner reads as
plain dependency injection.

---

## Task 4 — Refactor IOManager

### 4a. Drop the unjustified `IOManager` ABC

With a single implementation, the ABC is indirection. Remove the abstract base
in `io_utils/io_base.py` and make `LocalFSIOManager` a standalone concrete
class. Update the two importers (`runner.py`, and the now-deleted factories) to
import/type-hint `LocalFSIOManager` directly. Delete `io_base.py` once nothing
imports `IOManager`.

### 4b. Split `save_step()` into helpers + `match`/`case`

`save_step` is one long `if/elif/elif/else` on `output_type` (it even carries a
`# TODO` to do exactly this). Extract one private helper per branch and dispatch
with `match`:

```python
def save_step(self, time, step, state, laser_path, **kwargs):
    output_type = kwargs.get("output_type", "full_volume")
    try:
        match output_type:
            case "full_volume": self._save_full_volume(time, step, state)
            case "modes":       self._save_modes(time, step, state)
            case "profiles":    self._save_profiles(time, step, state, laser_path, kwargs.get("profiles_locations", []))
            case "cut_views":   self._save_cut_views(time, step, state, laser_path, kwargs.get("cut_views_planes", []))
            case _:             logger.warning("Unknown output_type '%s' in save_step. Skipping.", output_type)
    except Exception as e:
        logger.error("Failed to save %s for step %s: %s", output_type, step, e)
        raise
```

New private methods (logic moved verbatim, not rewritten): `_save_full_volume`,
`_save_modes`, `_save_profiles`, `_save_cut_views`. The shared try/except and
the lazy `reconstruct_temperature_DCT` import stay in the relevant scope. Removes
the standing `# TODO`.

---

## Validation

- `pytest tests/test_integration.py -k cpu` (CPU e2e) must pass after the changes
  — it exercises runner construction, `build_solver`, and `save_step`'s
  full_volume path end to end.
- `grep` to confirm no remaining references to `SimulationFactory`,
  `*SimulationFactory`, or `IOManager` (the ABC) anywhere in `src/`, `tests/`,
  `simulations/`.

## Order of work

1. Task 5 first (factories → `build_solver` + runner DI) — it changes the runner
   signature that everything else depends on.
2. Task 4a (drop ABC) — small, unblocks clean imports.
3. Task 4b (`save_step` split) — self-contained, no signature changes.
4. Run validation.
