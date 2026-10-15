"""Tests for the single-temperature simulation workflow."""

import numpy as np
import pytest

from isinglab.model import IsingModel
from isinglab.simulation import SingleTemperatureSimulation, run_simulation


def test_equilibration_is_discarded_and_measurements_have_expected_length() -> None:
    result = run_simulation(
        L=4,
        temperature=2.0,
        seed=10,
        equilibration_sweeps=3,
        measurement_sweeps=5,
    )

    assert result.equilibration_sweeps == 3
    assert result.measurement_sweeps == 5
    assert len(result.energies) == 5
    assert len(result.magnetizations) == 5
    assert result.attempted_flips == 8 * 16


def test_result_reports_configuration_and_acceptance_statistics() -> None:
    result = run_simulation(
        L=3,
        J=1.5,
        h=-0.25,
        temperature=2.25,
        k_B=0.5,
        initialization="ordered",
        seed=4,
        equilibration_sweeps=1,
        measurement_sweeps=2,
    )

    assert result.configuration == {
        "L": 3,
        "J": 1.5,
        "h": -0.25,
        "temperature": 2.25,
        "k_B": 0.5,
        "initialization": "ordered",
    }
    assert result.accepted_flips <= result.attempted_flips
    assert 0 <= result.acceptance_rate <= 1
    assert result.acceptance_rate == pytest.approx(
        result.accepted_flips / result.attempted_flips
    )


@pytest.mark.parametrize("initialization", ["random", "ordered"])
def test_constructed_model_reports_initialization(initialization: str) -> None:
    result = run_simulation(
        L=3,
        temperature=2.0,
        initialization=initialization,
        seed=5,
    )

    assert result.configuration["initialization"] == initialization


def test_external_model_reports_external_initialization() -> None:
    model = IsingModel(3, initialization="ordered")

    result = run_simulation(
        model=model,
        temperature=2.0,
        measurement_sweeps=1,
    )

    assert result.configuration["initialization"] == "external"


def test_equilibration_sweeps_are_not_recorded() -> None:
    model = IsingModel(2, initialization="ordered")
    simulation = SingleTemperatureSimulation(
        model=model,
        temperature=1.0,
        equilibration_sweeps=2,
        measurement_sweeps=1,
    )

    result = simulation.run()

    assert result.energies.shape == (1,)
    assert result.magnetizations.shape == (1,)
    assert result.attempted_flips == 3 * model.L**2


def test_fixed_seed_reproduces_configuration_samples_and_observables() -> None:
    arguments = {
        "L": 5,
        "temperature": 2.1,
        "seed": 123,
        "equilibration_sweeps": 2,
        "measurement_sweeps": 4,
    }

    first = run_simulation(**arguments)
    second = run_simulation(**arguments)

    assert np.array_equal(first.energies, second.energies)
    assert np.array_equal(first.magnetizations, second.magnetizations)
    assert first.observables == second.observables
    assert first.accepted_flips == second.accepted_flips


def test_existing_model_and_metropolis_can_be_supplied() -> None:
    model = IsingModel(4, initialization="ordered", seed=7)
    simulation = SingleTemperatureSimulation(
        model=model,
        temperature=2.0,
        equilibration_sweeps=0,
        measurement_sweeps=2,
    )

    result = simulation.run()

    assert result.configuration["L"] == 4
    assert len(result.energies) == 2


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("equilibration_sweeps", -1),
        ("measurement_sweeps", 0),
        ("equilibration_sweeps", 1.5),
        ("measurement_sweeps", True),
    ],
)
def test_invalid_sweep_counts_are_rejected(name: str, value: object) -> None:
    kwargs = {"L": 4, "temperature": 2.0, name: value}

    with pytest.raises(ValueError):
        SingleTemperatureSimulation(**kwargs)  # type: ignore[arg-type]