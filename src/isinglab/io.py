"""Safe, readable data export for Ising simulations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .experiments import TemperatureSweepResult
from .model import IsingModel
from .simulation import SimulationResult


def export_temperature_sweep_csv(
    result: TemperatureSweepResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export temperature-sweep observables as one row per temperature."""
    _validate_result(result)
    destination = _prepare_path(path, ".csv", overwrite)
    fields = ["temperature", *result.observables.keys()]
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, temperature in enumerate(result.temperatures):
            writer.writerow(
                {
                    "temperature": float(temperature),
                    **{
                        name: float(values[index])
                        for name, values in result.observables.items()
                    },
                }
            )
    return destination


def export_simulation_json(
    result: SimulationResult | TemperatureSweepResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export simulation or sweep configuration and metadata as JSON."""
    if not isinstance(result, (SimulationResult, TemperatureSweepResult)):
        raise TypeError("result must be a SimulationResult or TemperatureSweepResult")
    destination = _prepare_path(path, ".json", overwrite)
    payload = {
        "configuration": _json_value(result.configuration),
        "metadata": _metadata(result),
    }
    if isinstance(result, SimulationResult):
        payload["observables"] = _json_value(result.observables)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return destination


def save_spin_configuration(
    spins: IsingModel | np.ndarray,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a validated spin array in NumPy ``.npy`` format."""
    values = spins.spins if isinstance(spins, IsingModel) else np.asarray(spins)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isin(values, [-1, 1])):
        raise ValueError("spins must be a non-empty two-dimensional array of -1 and +1")
    destination = _prepare_path(path, ".npy", overwrite)
    np.save(destination, values, allow_pickle=False)
    return destination


export_spin_configuration = save_spin_configuration


def _prepare_path(path: str | Path, suffix: str, overwrite: bool) -> Path:
    destination = Path(path)
    if not destination.name or destination.exists() and destination.is_dir():
        raise ValueError("path must identify a file, not a directory")
    if destination.suffix.lower() != suffix:
        raise ValueError(f"path must use the {suffix} extension")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _validate_result(result: TemperatureSweepResult) -> None:
    if not isinstance(result, TemperatureSweepResult):
        raise TypeError("result must be a TemperatureSweepResult")
    count = result.temperatures.size
    if count == 0 or any(values.shape != (count,) for values in result.observables.values()):
        raise ValueError("temperature-sweep arrays must be non-empty and equally sized")
    if not np.all(np.isfinite(result.temperatures)):
        raise ValueError("temperature values must be finite")


def _metadata(result: SimulationResult | TemperatureSweepResult) -> dict[str, Any]:
    if isinstance(result, TemperatureSweepResult):
        return {
            "kind": "temperature_sweep",
            "temperature_count": int(result.temperatures.size),
            "observables": list(result.observables),
        }
    return {
        "kind": "single_temperature_simulation",
        "measurement_count": int(result.energies.size),
        "attempted_flips": result.attempted_flips,
        "accepted_flips": result.accepted_flips,
        "acceptance_rate": result.acceptance_rate,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value