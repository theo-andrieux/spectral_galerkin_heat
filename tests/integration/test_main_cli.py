"""simulations/main.py: run directory, visualisation hook and opt-in profiling."""

import importlib.util
from pathlib import Path

import pytest
import yaml

MAIN = Path(__file__).resolve().parents[2] / "simulations" / "main.py"


def _load_main():
    spec = importlib.util.spec_from_file_location("sgh_main_cli", MAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_case(tmp_path):
    (tmp_path / "paths").mkdir()
    (tmp_path / "paths" / "scan.gcode").write_text(
        "G21\nG0 X0.1 Y0.1\nM3 S50.0\nG1 X0.3 Y0.1 F6000.0\n")
    cfg = {
        "simulation": {"backend": "cpu_linear", "duration": 2e-5, "dt": 2e-6},
        "domain": {"size": [4e-4, 2e-4, 1e-4], "mesh": [16, 8, 8]},
        "material": {"name": "316L", "rho": 7957.5, "k": 13.851, "Cp": 497.89, "T0": 293.0},
        "laser": {"radius": 30e-6, "absorptivity": 0.3, "power_nominal": 50.0,
                  "path": {"type": "gcode", "file": "scan.gcode",
                           "initial_position": [0.1, 0.1]}},
        "io": {"output_root": str(tmp_path / "out"), "run_tag": "cli",
               "output_interval": None, "at_end": ["full_volume"]},
    }
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.mark.integration
def test_main_returns_run_dir_and_runs_visualisation(tmp_path, monkeypatch, capsys):
    main_mod = _load_main()
    calls = []
    monkeypatch.setattr(main_mod, "generate_plots", lambda **kw: calls.append(kw))
    monkeypatch.setattr("sys.argv", ["main.py", str(_write_case(tmp_path)), "--viz"])

    run_dir = main_mod.main()

    assert run_dir is not None and Path(run_dir).is_dir()
    assert len(calls) == 1 and calls[0]["run_dir"] == run_dir
    assert not (Path(run_dir) / "diagnostics" / "profiler.txt").exists()
    assert "cumtime" not in capsys.readouterr().out


@pytest.mark.integration
def test_profile_flag_writes_report_to_run_dir(tmp_path, monkeypatch):
    main_mod = _load_main()
    monkeypatch.setattr("sys.argv",
                        ["main.py", str(_write_case(tmp_path)), "--no-viz", "--profile"])

    run_dir = main_mod.main_profiled()

    report = Path(run_dir) / "diagnostics" / "profiler.txt"
    assert report.is_file() and "cumtime" in report.read_text()
