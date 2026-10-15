"""Tests for thermodynamic measurements and statistical observables."""

import numpy as np
import pytest

from isinglab.model import IsingModel
from isinglab.observables import (
    MeasurementSeries,
    calculate_observables,
    measure_model,
)


def test_measure_model_reports_total_and_per_spin_values() -> None:
    model = IsingModel(4, initialization="ordered")

    measurement = measure_model(model)

    assert measurement["energy"] == -32
    assert measurement["energy_per_spin"] == -2
    assert measurement["magnetization"] == 16
    assert measurement["magnetization_per_spin"] == 1
    assert measurement["absolute_magnetization_per_spin"] == 1


def test_measurement_series_collects_numpy_arrays() -> None:
    model = IsingModel(2, initialization="ordered")
    measurements = MeasurementSeries()

    measurements.record(model)
    model.flip_spin(0, 0)
    measurements.record(model)

    assert isinstance(measurements.energies, np.ndarray)
    assert isinstance(measurements.magnetizations, np.ndarray)
    assert np.array_equal(measurements.energies, [-8, 0])
    assert np.array_equal(measurements.magnetizations, [4, 2])


def test_means_and_absolute_magnetization_are_normalized_by_n() -> None:
    result = calculate_observables(
        energies=[-8, 0, 8, 0],
        magnetizations=[4, 2, -2, -4],
        L=2,
        temperature=2,
    )

    assert result["mean_energy"] == 0
    assert result["mean_energy_per_spin"] == 0
    assert result["mean_magnetization"] == 0
    assert result["mean_magnetization_per_spin"] == 0
    assert result["mean_absolute_magnetization_per_spin"] == 0.75


def test_specific_heat_fluctuation_formula() -> None:
    result = calculate_observables(
        energies=[-2, 0, 2],
        magnetizations=[0, 0, 0],
        L=2,
        temperature=2,
        k_B=1,
    )

    # Var(H) = 8/3; C_V/N = (8/3) / (4 * 2^2).
    assert result["specific_heat_per_spin"] == pytest.approx(1 / 6)


def test_fluctuation_formulas_are_stable_for_large_common_offsets() -> None:
    result = calculate_observables(
        energies=[1e12 - 1, 1e12 + 1],
        magnetizations=[-1e12 - 2, -1e12 + 2],
        L=2,
        temperature=1,
        k_B=1,
    )

    assert result["specific_heat_per_spin"] == pytest.approx(0.25)
    assert result["susceptibility_per_spin"] == pytest.approx(1.0)


def test_susceptibility_uses_signed_magnetization_formula() -> None:
    result = calculate_observables(
        energies=[0, 0, 0],
        magnetizations=[-4, 0, 4],
        L=2,
        temperature=2,
        k_B=1,
    )

    # Var(M) = 32/3; chi/N = (32/3) / (4 * 2).
    assert result["susceptibility_per_spin"] == pytest.approx(4 / 3)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_is_rejected(temperature: float) -> None:
    with pytest.raises(ValueError):
        calculate_observables([0], [0], L=2, temperature=temperature)


def test_empty_samples_are_rejected() -> None:
    with pytest.raises(ValueError):
        calculate_observables([], [], L=2, temperature=1)


def test_mismatched_samples_are_rejected() -> None:
    with pytest.raises(ValueError):
        calculate_observables([0, 1], [0], L=2, temperature=1)