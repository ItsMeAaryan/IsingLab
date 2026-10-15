"""Tests for Metropolis Monte Carlo updates."""

import numpy as np
import pytest

from isinglab.metropolis import Metropolis
from isinglab.model import IsingModel


class FixedRng:
    """Small deterministic RNG double for selecting one lattice site."""

    def __init__(self, random_value: float) -> None:
        self.random_value = random_value

    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        return np.array([0, 0])

    def random(self) -> float:
        return self.random_value


def test_energy_lowering_flip_is_accepted() -> None:
    model = IsingModel(4, initialization="ordered", seed=1)
    model.spins[0, 0] = -1
    model.rng = FixedRng(random_value=1.0)  # type: ignore[assignment]
    sampler = Metropolis(model, temperature=1.0)

    assert sampler.attempt_flip() is True
    assert model.spins[0, 0] == 1
    assert sampler.attempted_flips == 1
    assert sampler.accepted_flips == 1


def test_energy_increasing_flip_uses_boltzmann_probability() -> None:
    model = IsingModel(4, initialization="ordered")
    model.rng = FixedRng(random_value=0.1)  # type: ignore[assignment]
    sampler = Metropolis(model, temperature=2.0, k_B=1.0)

    # Ordered state: flipping one spin costs 8, and exp(-8 / 2) < 0.1.
    assert sampler.attempt_flip() is False
    assert sampler.acceptance_rate == 0.0


def test_energy_increasing_flip_can_be_accepted() -> None:
    model = IsingModel(4, initialization="ordered")
    model.rng = FixedRng(random_value=0.01)  # below exp(-8 / 2)
    sampler = Metropolis(model, temperature=2.0)

    assert sampler.attempt_flip() is True


def test_energy_increasing_flip_at_exact_probability_threshold_is_rejected() -> None:
    model = IsingModel(4, initialization="ordered")
    model.rng = FixedRng(random_value=float(np.exp(-8 / 2)))
    sampler = Metropolis(model, temperature=2.0)

    assert sampler.attempt_flip() is False


def test_zero_or_negative_temperature_and_kb_are_rejected() -> None:
    model = IsingModel(4, initialization="ordered")

    with pytest.raises(ValueError):
        Metropolis(model, temperature=0)
    with pytest.raises(ValueError):
        Metropolis(model, temperature=-1)
    with pytest.raises(ValueError):
        Metropolis(model, temperature=1, k_B=0)
    with pytest.raises(ValueError):
        Metropolis(model, temperature=1, k_B=-1)


def test_sweep_performs_exactly_l_squared_attempts() -> None:
    sampler = Metropolis(IsingModel(5, seed=12), temperature=2.0)

    sampler.sweep()

    assert sampler.attempted_flips == 25


def test_acceptance_counters_and_rate_are_correct() -> None:
    model = IsingModel(4, initialization="ordered", seed=1)
    model.spins[0, 0] = -1
    model.rng = FixedRng(random_value=0.99)  # type: ignore[assignment]
    sampler = Metropolis(model, temperature=1.0)

    assert sampler.attempt_flip() is True
    assert sampler.attempt_flip() is False
    assert sampler.attempted_flips == 2
    assert sampler.accepted_flips == 1
    assert sampler.acceptance_rate == pytest.approx(0.5)


def test_identical_seeds_produce_reproducible_results() -> None:
    first = Metropolis(IsingModel(6, seed=42), temperature=2.25)
    second = Metropolis(IsingModel(6, seed=42), temperature=2.25)

    first.sweep()
    second.sweep()

    assert np.array_equal(first.model.spins, second.model.spins)
    assert first.attempted_flips == second.attempted_flips
    assert first.accepted_flips == second.accepted_flips