"""Thermodynamic measurements and fluctuation observables.

The fluctuation formulas in this module use *total* energy ``H`` and
*signed total* magnetization ``M`` samples.  Results labelled "per spin"
are normalized by ``N = L**2`` only after the relevant total quantity has
been calculated.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from .model import IsingModel


@dataclass
class MeasurementSeries:
    """Collect total energy and signed total magnetization measurements."""

    energies: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    magnetizations: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.energies = _validate_samples(self.energies, "energies", allow_empty=True)
        self.magnetizations = _validate_samples(
            self.magnetizations, "magnetizations", allow_empty=True
        )
        if self.energies.size != self.magnetizations.size:
            raise ValueError("energies and magnetizations must have equal lengths")

    def record(self, model: IsingModel) -> None:
        """Append one measurement from the current model state."""
        if not isinstance(model, IsingModel):
            raise TypeError("model must be an IsingModel instance")
        self.energies = np.append(self.energies, model.energy())
        self.magnetizations = np.append(self.magnetizations, model.magnetization())

    def __len__(self) -> int:
        return self.energies.size


def measure_model(model: IsingModel) -> dict[str, float]:
    """Return instantaneous total and per-spin measurements."""
    if not isinstance(model, IsingModel):
        raise TypeError("model must be an IsingModel instance")
    total_energy = model.energy()
    total_magnetization = model.magnetization()
    n_spins = model.L * model.L
    return {
        "energy": total_energy,
        "energy_per_spin": total_energy / n_spins,
        "magnetization": float(total_magnetization),
        "magnetization_per_spin": total_magnetization / n_spins,
        "absolute_magnetization_per_spin": abs(total_magnetization) / n_spins,
    }


def calculate_observables(
    energies: Iterable[float] | NDArray[np.float64],
    magnetizations: Iterable[float] | NDArray[np.float64],
    L: int,
    temperature: float,
    k_B: float = 1.0,
) -> dict[str, float]:
    """Calculate means and fluctuation observables from total samples.

    ``magnetizations`` are signed total magnetizations.  Consequently,
    susceptibility uses ``<M**2> - <M>**2`` rather than absolute
    magnetization.  Absolute magnetization is reported separately as
    ``<|M|>/N``.
    """
    if isinstance(L, bool) or not isinstance(L, Integral) or L < 2:
        raise ValueError("L must be an integer greater than or equal to 2")
    if not isinstance(temperature, Real) or not np.isfinite(temperature):
        raise ValueError("temperature must be a finite positive number")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not isinstance(k_B, Real) or not np.isfinite(k_B) or k_B <= 0:
        raise ValueError("k_B must be positive and finite")

    energy_samples = _validate_samples(energies, "energies")
    magnetization_samples = _validate_samples(magnetizations, "magnetizations")
    if energy_samples.size != magnetization_samples.size:
        raise ValueError("energies and magnetizations must have equal lengths")

    n_spins = int(L) * int(L)
    mean_energy = float(np.mean(energy_samples))
    mean_magnetization = float(np.mean(magnetization_samples))
    energy_variance = float(np.var(energy_samples, ddof=0))
    magnetization_variance = float(np.var(magnetization_samples, ddof=0))

    return {
        "mean_energy": mean_energy,
        "mean_energy_per_spin": mean_energy / n_spins,
        "mean_magnetization": mean_magnetization,
        "mean_magnetization_per_spin": mean_magnetization / n_spins,
        "mean_absolute_magnetization_per_spin": float(
            np.mean(np.abs(magnetization_samples)) / n_spins
        ),
        "specific_heat_per_spin": (
            energy_variance
        )
        / (n_spins * float(k_B) * float(temperature) ** 2),
        "susceptibility_per_spin": magnetization_variance
        / (n_spins * float(k_B) * float(temperature)),
    }


def _validate_samples(
    samples: Iterable[float] | NDArray[np.float64],
    name: str,
    allow_empty: bool = False,
) -> NDArray[np.float64]:
    values = np.asarray(list(samples) if not isinstance(samples, np.ndarray) else samples)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sample array")
    if values.size == 0 and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    try:
        numeric = values.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numbers") from error
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain finite numbers")
    return numeric.copy()