"""Tests for non-interactive visualizations."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from isinglab.experiments import run_temperature_sweep
from isinglab.model import IsingModel
from isinglab.visualization import (
    plot_absolute_magnetization_per_spin,
    plot_energy_per_spin,
    plot_magnetization_per_spin,
    plot_spin_configuration,
    plot_specific_heat,
    plot_susceptibility,
)


def test_visualization_functions_return_axes() -> None:
    result = run_temperature_sweep([1.5, 2.5], L=3, seed=1)

    axes = [
        plot_spin_configuration(IsingModel(3, initialization="ordered")),
        plot_energy_per_spin(result),
        plot_magnetization_per_spin(result),
        plot_absolute_magnetization_per_spin(result),
        plot_specific_heat(result),
        plot_susceptibility(result),
    ]

    assert all(axis.figure is not None for axis in axes)
    assert any("critical" in text.get_text().lower() for text in axes[1].get_legend().get_texts())
    plt.close("all")


def test_spin_heatmap_accepts_numpy_array() -> None:
    axis = plot_spin_configuration(np.ones((2, 2), dtype=np.int8))
    assert axis.images[0].get_array().shape == (2, 2)
    plt.close(axis.figure)
