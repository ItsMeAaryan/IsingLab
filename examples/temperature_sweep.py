"""Run a small temperature sweep around the infinite-lattice critical region.

Run from the project root with:

    python examples/temperature_sweep.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from isinglab.experiments import run_temperature_sweep
from isinglab.io import export_simulation_json, export_temperature_sweep_csv
from isinglab.visualization import (
    plot_energy_per_spin,
    plot_magnetization_per_spin,
    plot_specific_heat,
    plot_susceptibility,
)


def main() -> None:
    output_dir = Path("examples") / "output" / "temperature_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_temperature_sweep(
        temperatures=[1.8, 2.1, 2.269185, 2.5, 2.8],
        L=16,
        equilibration_sweeps=25,
        measurement_sweeps=50,
        seed=2026,
    )

    export_temperature_sweep_csv(
        result, output_dir / "temperature_sweep.csv", overwrite=True
    )
    export_simulation_json(
        result, output_dir / "configuration.json", overwrite=True
    )

    plots = {
        "energy": plot_energy_per_spin(result),
        "magnetization": plot_magnetization_per_spin(result),
        "specific_heat": plot_specific_heat(result),
        "susceptibility": plot_susceptibility(result),
    }
    for name, axis in plots.items():
        axis.figure.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(axis.figure)

    print(f"Saved sweep data and plots to {output_dir}")
    print(
        "Reference infinite-lattice critical temperature: "
        f"{result.configuration['critical_temperature_reference']:.6f}"
    )


if __name__ == "__main__":
    main()