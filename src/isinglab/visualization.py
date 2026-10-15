"""Matplotlib visualizations for Ising models and temperature sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .experiments import (
    INFINITE_LATTICE_CRITICAL_TEMPERATURE,
    TemperatureSweepResult,
)
from .model import IsingModel


def plot_spin_configuration(
    spins: IsingModel | np.ndarray,
    *,
    ax: Any = None,
    title: str = "Spin configuration",
) -> Any:
    """Plot a spin configuration as a ``-1``/``+1`` heatmap."""
    values = _spin_array(spins)
    axis = _axis(ax)
    image = axis.imshow(values, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    axis.set_title(title)
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    axis.figure.colorbar(image, ax=axis, ticks=[-1, 1], label="Spin")
    return axis


def plot_temperature_observable(
    result: TemperatureSweepResult,
    observable: str,
    *,
    ax: Any = None,
    title: str | None = None,
    ylabel: str | None = None,
) -> Any:
    """Plot one temperature-dependent observable from a sweep result."""
    if not isinstance(result, TemperatureSweepResult):
        raise TypeError("result must be a TemperatureSweepResult")
    labels = {
        "mean_energy_per_spin": ("Energy per spin", "Mean energy per spin"),
        "mean_magnetization_per_spin": (
            "Magnetization per spin",
            "Mean magnetization per spin",
        ),
        "mean_absolute_magnetization_per_spin": (
            "Absolute magnetization per spin",
            "Mean absolute magnetization per spin",
        ),
        "specific_heat_per_spin": ("Specific heat per spin", "Specific heat per spin"),
        "susceptibility_per_spin": (
            "Magnetic susceptibility per spin",
            "Magnetic susceptibility per spin",
        ),
    }
    if observable not in labels:
        raise ValueError(f"unknown observable: {observable}")

    axis = _axis(ax)
    axis.plot(result.temperatures, result.observables[observable], marker="o", label=labels[observable][0])
    axis.set_title(title or labels[observable][1])
    axis.set_xlabel("Temperature (T)")
    axis.set_ylabel(ylabel or labels[observable][0])
    if _has_reference_critical_temperature(result):
        axis.axvline(
            INFINITE_LATTICE_CRITICAL_TEMPERATURE,
            color="black",
            linestyle="--",
            label="Infinite-lattice reference critical temperature $T_c$",
        )
    axis.legend()
    return axis


def plot_energy_per_spin(result: TemperatureSweepResult, *, ax: Any = None) -> Any:
    """Plot mean energy per spin versus temperature."""
    return plot_temperature_observable(result, "mean_energy_per_spin", ax=ax)


def plot_magnetization_per_spin(result: TemperatureSweepResult, *, ax: Any = None) -> Any:
    """Plot mean signed magnetization per spin versus temperature."""
    return plot_temperature_observable(result, "mean_magnetization_per_spin", ax=ax)


def plot_absolute_magnetization_per_spin(
    result: TemperatureSweepResult, *, ax: Any = None
) -> Any:
    """Plot mean absolute magnetization per spin versus temperature."""
    return plot_temperature_observable(
        result, "mean_absolute_magnetization_per_spin", ax=ax
    )


def plot_specific_heat(result: TemperatureSweepResult, *, ax: Any = None) -> Any:
    """Plot specific heat per spin versus temperature."""
    return plot_temperature_observable(result, "specific_heat_per_spin", ax=ax)


def plot_susceptibility(result: TemperatureSweepResult, *, ax: Any = None) -> Any:
    """Plot magnetic susceptibility per spin versus temperature."""
    return plot_temperature_observable(result, "susceptibility_per_spin", ax=ax)


def _axis(ax: Any) -> Any:
    return plt.subplots()[1] if ax is None else ax


def _spin_array(spins: IsingModel | np.ndarray) -> np.ndarray:
    values = spins.spins if isinstance(spins, IsingModel) else np.asarray(spins)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isin(values, [-1, 1])):
        raise ValueError("spins must be a non-empty two-dimensional array of -1 and +1")
    return values


def _has_reference_critical_temperature(result: TemperatureSweepResult) -> bool:
    return (
        result.configuration.get("J") == 1.0
        and result.configuration.get("h") == 0.0
    )