"""Tests for data exports."""

import json

import numpy as np
import pytest

from isinglab.experiments import run_temperature_sweep
from isinglab.io import (
    export_simulation_json,
    export_temperature_sweep_csv,
    save_spin_configuration,
)
from isinglab.model import IsingModel
from isinglab.simulation import run_simulation


def test_temperature_sweep_csv_and_json_exports(tmp_path) -> None:
    result = run_temperature_sweep([1.5, 2.5], L=3, seed=4)
    csv_path = export_temperature_sweep_csv(result, tmp_path / "nested" / "results.csv")
    json_path = export_simulation_json(result, tmp_path / "nested" / "metadata.json")

    assert csv_path.read_text(encoding="utf-8").count("\n") == 3
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["kind"] == "temperature_sweep"
    assert payload["configuration"]["L"] == 3


def test_spin_configuration_is_saved_as_npy(tmp_path) -> None:
    model = IsingModel(3, initialization="ordered")
    path = save_spin_configuration(model, tmp_path / "spins.npy")

    assert np.array_equal(np.load(path), model.spins)


def test_json_export_supports_single_simulation(tmp_path) -> None:
    result = run_simulation(L=3, temperature=2, seed=1)
    payload_path = export_simulation_json(result, tmp_path / "result.json")
    payload = json.loads(payload_path.read_text())
    assert payload["metadata"]["kind"] == "single_temperature_simulation"
    assert payload["observables"] == result.observables
    assert payload["metadata"]["measurement_count"] == result.measurement_sweeps
    assert payload["metadata"]["attempted_flips"] == result.attempted_flips


def test_exports_refuse_overwrite_and_invalid_inputs(tmp_path) -> None:
    result = run_temperature_sweep([2.0], L=3, seed=1)
    path = export_temperature_sweep_csv(result, tmp_path / "result.csv")
    with pytest.raises(FileExistsError):
        export_temperature_sweep_csv(result, path)
    with pytest.raises(ValueError):
        export_temperature_sweep_csv(result, tmp_path / "result.txt")
    with pytest.raises(ValueError):
        save_spin_configuration(np.zeros((2, 2)), tmp_path / "bad.npy")
