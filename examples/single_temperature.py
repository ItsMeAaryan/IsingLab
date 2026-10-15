"""Run and visualize one small Ising simulation.

Run from the project root with:

    python examples/single_temperature.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from isinglab.io import export_simulation_json
from isinglab.simulation import SingleTemperatureSimulation
from isinglab.visualization import plot_spin_configuration


def main() -> None:
    output_dir = Path("examples") / "output" / "single_temperature"
    output_dir.mkdir(parents=True, exist_ok=True)

    simulation = SingleTemperatureSimulation(
        L=16,
        temperature=2.269185,
        equilibration_sweeps=25,
        measurement_sweeps=50,
        seed=12345,
    )
    result = simulation.run()

    export_simulation_json(result, output_dir / "results.json", overwrite=True)
    axis = plot_spin_configuration(simulation.model)
    axis.figure.savefig(output_dir / "spin_configuration.png", dpi=150, bbox_inches="tight")
    plt.close(axis.figure)

    print(f"Saved results and spin plot to {output_dir}")
    print(f"Mean energy per spin: {result.observables['mean_energy_per_spin']:.6f}")
    print(
        "Mean absolute magnetization per spin: "
        f"{result.observables['mean_absolute_magnetization_per_spin']:.6f}"
    )
    print(f"Acceptance rate: {result.acceptance_rate:.3f}")


if __name__ == "__main__":
    main()