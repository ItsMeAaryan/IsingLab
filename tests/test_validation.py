"""Validation and temperature-sweep tests."""

import numpy as np
import pytest

from isinglab.experiments import (
    INFINITE_LATTICE_CRITICAL_TEMPERATURE,
    run_temperature_sweep,
)


def test_temperature_sweep_returns_all_observables() -> None:
    result = run_temperature_sweep(
        [1.5, 2.269185, 3.0],
        L=4,
        seed=11,
        equilibration_sweeps=1,
        measurement_sweeps=3,
    )

    assert np.array_equal(result.temperatures, [1.5, 2.269185, 3.0])
    assert result.mean_energy_per_spin.shape == (3,)
    assert result.mean_magnetization_per_spin.shape == (3,)
    assert result.mean_absolute_magnetization_per_spin.shape == (3,)
    assert result.specific_heat_per_spin.shape == (3,)
    assert result.susceptibility_per_spin.shape == (3,)
    assert set(result.observables) == {
        "mean_energy_per_spin",
        "mean_magnetization_per_spin",
        "mean_absolute_magnetization_per_spin",
        "specific_heat_per_spin",
        "susceptibility_per_spin",
    }


def test_temperature_sweep_is_reproducible_and_uses_independent_seeds() -> None:
    arguments = {
        "temperatures": np.array([1.0, 2.0, 3.0]),
        "L": 4,
        "seed": 42,
        "equilibration_sweeps": 1,
        "measurement_sweeps": 2,
    }

    first = run_temperature_sweep(**arguments)
    second = run_temperature_sweep(**arguments)

    assert np.array_equal(first.mean_energy_per_spin, second.mean_energy_per_spin)
    assert np.array_equal(first.susceptibility_per_spin, second.susceptibility_per_spin)
    assert first.configuration["child_seeds"] == second.configuration["child_seeds"]
    assert len(set(first.configuration["child_seeds"])) == 3


def test_critical_temperature_reference_is_documented() -> None:
    assert INFINITE_LATTICE_CRITICAL_TEMPERATURE == pytest.approx(2.269185, abs=1e-6)


@pytest.mark.parametrize(
    "temperatures",
    [[], np.array([]), [0.0, 1.0], [-1.0], [np.nan], [np.inf], [[1.0, 2.0]]],
)
def test_invalid_temperature_arrays_are_rejected(temperatures: object) -> None:
    with pytest.raises(ValueError):
        run_temperature_sweep(temperatures, L=4)  # type: ignore[arg-type]


def test_invalid_simulation_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        run_temperature_sweep([1.0], L=1)
    with pytest.raises(ValueError):
        run_temperature_sweep([1.0], L=4, measurement_sweeps=0)
    with pytest.raises(ValueError):
        run_temperature_sweep([1.0], L=4, equilibration_sweeps=-1)
    with pytest.raises(ValueError):
        run_temperature_sweep([1.0], L=4, seed=1.5)  # type: ignore[arg-type]