"""Tests for the mathematical Ising model core."""

import numpy as np
import pytest
from itertools import product

from isinglab.model import IsingModel


def test_ordered_four_by_four_energy_and_magnetization() -> None:
    model = IsingModel(4, J=1, h=0, initialization="ordered")

    assert model.energy() == -32
    assert model.magnetization() == 16
    assert model.magnetization_per_spin() == 1


def test_energy_matches_explicit_positive_direction_bond_reference() -> None:
    model = IsingModel(3, J=1.7, h=-0.25, seed=4)
    reference = 0.0
    for row in range(model.L):
        for column in range(model.L):
            reference -= model.J * model.spins[row, column] * (
                model.spins[row, (column + 1) % model.L]
                + model.spins[(row + 1) % model.L, column]
            )
            reference -= model.h * model.spins[row, column]

    assert model.energy() == pytest.approx(reference)


@pytest.mark.parametrize("configuration", list(product((-1, 1), repeat=4)))
def test_exhaustive_l2_energy_matches_explicit_bond_reference(
    configuration: tuple[int, ...],
) -> None:
    model = IsingModel(2, J=1.3, h=-0.4, initialization="ordered")
    model.spins[:] = np.asarray(configuration, dtype=np.int8).reshape(2, 2)
    reference = 0.0
    for row in range(2):
        for column in range(2):
            reference -= model.J * model.spins[row, column] * (
                model.spins[row, (column + 1) % 2]
                + model.spins[(row + 1) % 2, column]
            )
            reference -= model.h * model.spins[row, column]

    assert model.energy() == pytest.approx(reference)


def test_random_initialization_contains_only_valid_spins_and_is_reproducible() -> None:
    first = IsingModel(8, initialization="random", seed=123)
    second = IsingModel(8, initialization="random", seed=123)

    assert np.array_equal(first.spins, second.spins)
    assert set(np.unique(first.spins)).issubset({-1, 1})


def test_flip_changes_magnetization_by_two_times_the_spin() -> None:
    model = IsingModel(4, initialization="ordered")
    before = model.magnetization()

    model.flip_spin(1, 2)

    assert model.magnetization() == before - 2
    assert model.spins[1, 2] == -1


def test_local_energy_change_matches_full_energy_difference() -> None:
    model = IsingModel(5, J=1.7, h=-0.25, seed=7)
    before = model.energy()
    delta = model.local_energy_change(2, 4)

    model.flip_spin(2, 4)

    assert model.energy() - before == pytest.approx(delta)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_local_energy_change_matches_full_recomputation_for_all_sites(
    seed: int,
) -> None:
    model = IsingModel(4, J=0.8, h=0.3, seed=seed)
    for row in range(model.L):
        for column in range(model.L):
            before = model.energy()
            delta = model.flip_energy_change(row, column)
            model.flip_spin(row, column)
            assert model.energy() - before == pytest.approx(delta)
            model.flip_spin(row, column)


def test_periodic_boundaries_are_used_for_local_energy() -> None:
    model = IsingModel(4, J=1, h=0, initialization="ordered")
    model.spins[:, :] = 1
    model.spins[0, 0] = -1

    assert model.local_energy_change(0, 0) == -8


@pytest.mark.parametrize("size", [0, 1, -2, 2.5, True])
def test_invalid_lattice_sizes_are_rejected(size: object) -> None:
    with pytest.raises(ValueError):
        IsingModel(size)  # type: ignore[arg-type]


@pytest.mark.parametrize("initialization", ["invalid", "", None])
def test_invalid_initialization_methods_are_rejected(
    initialization: object,
) -> None:
    with pytest.raises(ValueError):
        IsingModel(4, initialization=initialization)  # type: ignore[arg-type]