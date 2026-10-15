"""Temperature-dependent Ising experiments.

For the infinite square-lattice ferromagnetic model with ``J=1`` and
``h=0``, the exact critical temperature is

.. math:: T_c = 2 / \\ln(1 + \\sqrt{2}) \\approx 2.269185.

This is a thermodynamic-limit reference value.  A finite lattice does not
reproduce it exactly: finite-size effects broaden and shift the transition,
and critical fluctuations require larger samples and lattices for reliable
estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from .simulation import run_simulation

INFINITE_LATTICE_CRITICAL_TEMPERATURE = float(
    2.0 / np.log(1.0 + np.sqrt(2.0))
)


@dataclass(frozen=True)
class TemperatureSweepResult:
    """Results from independent simulations at a sequence of temperatures."""

    temperatures: np.ndarray
    mean_energy_per_spin: np.ndarray
    mean_magnetization_per_spin: np.ndarray
    mean_absolute_magnetization_per_spin: np.ndarray
    specific_heat_per_spin: np.ndarray
    susceptibility_per_spin: np.ndarray
    configuration: dict[str, Any]
    results_by_temperature: tuple[Any, ...]

    @property
    def observables(self) -> dict[str, np.ndarray]:
        """Return all measured temperature-dependent observables."""
        return {
            "mean_energy_per_spin": self.mean_energy_per_spin,
            "mean_magnetization_per_spin": self.mean_magnetization_per_spin,
            "mean_absolute_magnetization_per_spin": (
                self.mean_absolute_magnetization_per_spin
            ),
            "specific_heat_per_spin": self.specific_heat_per_spin,
            "susceptibility_per_spin": self.susceptibility_per_spin,
        }


def run_temperature_sweep(
    temperatures: list[float] | tuple[float, ...] | np.ndarray,
    *,
    L: int,
    J: float = 1.0,
    h: float = 0.0,
    k_B: float = 1.0,
    initialization: str = "random",
    seed: int | None = None,
    equilibration_sweeps: int = 0,
    measurement_sweeps: int = 1,
) -> TemperatureSweepResult:
    """Run independent single-temperature simulations.

    Each temperature receives an independent child seed derived from
    ``seed`` and its position in the input sequence.  Thus repeated calls
    with the same inputs reproduce every result, while simulations at
    different temperatures do not share mutable model state.
    """
    temperature_values = _validate_temperatures(temperatures)
    _validate_seed(seed)

    child_seeds = _temperature_seeds(seed, temperature_values.size)
    runs = tuple(
        run_simulation(
            L=L,
            J=J,
            h=h,
            temperature=float(temperature),
            k_B=k_B,
            initialization=initialization,
            seed=child_seed,
            equilibration_sweeps=equilibration_sweeps,
            measurement_sweeps=measurement_sweeps,
        )
        for temperature, child_seed in zip(temperature_values, child_seeds)
    )

    def values(name: str) -> np.ndarray:
        return np.asarray([run.observables[name] for run in runs], dtype=np.float64)

    return TemperatureSweepResult(
        temperatures=temperature_values.copy(),
        mean_energy_per_spin=values("mean_energy_per_spin"),
        mean_magnetization_per_spin=values("mean_magnetization_per_spin"),
        mean_absolute_magnetization_per_spin=values(
            "mean_absolute_magnetization_per_spin"
        ),
        specific_heat_per_spin=values("specific_heat_per_spin"),
        susceptibility_per_spin=values("susceptibility_per_spin"),
        configuration={
            "L": int(L),
            "J": float(J),
            "h": float(h),
            "k_B": float(k_B),
            "initialization": initialization,
            "seed": seed,
            "child_seeds": child_seeds,
            "equilibration_sweeps": equilibration_sweeps,
            "measurement_sweeps": measurement_sweeps,
            "critical_temperature_reference": INFINITE_LATTICE_CRITICAL_TEMPERATURE,
        },
        results_by_temperature=runs,
    )


temperature_sweep = run_temperature_sweep


def _validate_temperatures(
    temperatures: list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(temperatures)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("temperatures must be a non-empty one-dimensional array")
    try:
        values = values.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("temperatures must contain finite positive numbers") from error
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("temperatures must contain finite positive numbers")
    return values


def _validate_seed(seed: int | None) -> None:
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, Integral)
    ):
        raise ValueError("seed must be an integer or None")


def _temperature_seeds(seed: int | None, count: int) -> list[int]:
    sequence = np.random.SeedSequence(seed)
    return [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in sequence.spawn(count)
    ]