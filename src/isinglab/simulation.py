"""Single-temperature Ising simulation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from .metropolis import Metropolis
from .model import IsingModel
from .observables import MeasurementSeries, calculate_observables


@dataclass(frozen=True)
class SimulationResult:
    """Results from one single-temperature simulation."""

    configuration: dict[str, Any]
    energies: np.ndarray
    magnetizations: np.ndarray
    observables: dict[str, float]
    attempted_flips: int
    accepted_flips: int
    acceptance_rate: float
    equilibration_sweeps: int
    measurement_sweeps: int


class SingleTemperatureSimulation:
    """Run equilibration and measurement sweeps at one temperature.

    Equilibration sweeps evolve the state toward thermal equilibrium but
    their measurements are deliberately discarded.  Measurement sweeps are
    then performed and one state measurement is recorded after each sweep.
    """

    def __init__(
        self,
        model: IsingModel | None = None,
        metropolis: Metropolis | None = None,
        *,
        L: int | None = None,
        J: float = 1.0,
        h: float = 0.0,
        temperature: float | None = None,
        k_B: float = 1.0,
        initialization: str = "random",
        seed: int | None = None,
        equilibration_sweeps: int = 0,
        measurement_sweeps: int = 1,
    ) -> None:
        self.equilibration_sweeps = _validate_sweep_count(
            equilibration_sweeps, "equilibration_sweeps"
        )
        self.measurement_sweeps = _validate_sweep_count(
            measurement_sweeps, "measurement_sweeps", minimum=1
        )
        initialization_metadata = "external"

        if metropolis is not None and not isinstance(metropolis, Metropolis):
            raise TypeError("metropolis must be a Metropolis instance")
        if model is not None and not isinstance(model, IsingModel):
            raise TypeError("model must be an IsingModel instance")
        if model is not None and metropolis is not None and metropolis.model is not model:
            raise ValueError("model and metropolis must refer to the same model")

        if metropolis is None:
            if model is None:
                if L is None:
                    raise ValueError("L is required when model is not provided")
                if temperature is None:
                    raise ValueError(
                        "temperature is required when metropolis is not provided"
                    )
                model = IsingModel(
                    L=L,
                    J=J,
                    h=h,
                    initialization=initialization,
                    seed=seed,
                )
                initialization_metadata = initialization
            elif temperature is None:
                raise ValueError(
                    "temperature is required when metropolis is not provided"
                )
            metropolis = Metropolis(model, temperature=temperature, k_B=k_B)
        elif model is None:
            model = metropolis.model

        self.model = model
        self.metropolis = metropolis
        self.initialization_metadata = initialization_metadata

    def run(self) -> SimulationResult:
        """Run the configured workflow and return structured results."""
        for _ in range(self.equilibration_sweeps):
            self.metropolis.sweep()

        measurements = MeasurementSeries()
        for _ in range(self.measurement_sweeps):
            self.metropolis.sweep()
            measurements.record(self.model)

        observables = calculate_observables(
            measurements.energies,
            measurements.magnetizations,
            L=self.model.L,
            temperature=self.metropolis.temperature,
            k_B=self.metropolis.k_B,
        )
        return SimulationResult(
            configuration={
                "L": self.model.L,
                "J": self.model.J,
                "h": self.model.h,
                "temperature": self.metropolis.temperature,
                "k_B": self.metropolis.k_B,
                "initialization": self.initialization_metadata,
            },
            energies=measurements.energies,
            magnetizations=measurements.magnetizations,
            observables=observables,
            attempted_flips=self.metropolis.attempted_flips,
            accepted_flips=self.metropolis.accepted_flips,
            acceptance_rate=self.metropolis.acceptance_rate,
            equilibration_sweeps=self.equilibration_sweeps,
            measurement_sweeps=self.measurement_sweeps,
        )


def run_simulation(
    model: IsingModel | None = None,
    metropolis: Metropolis | None = None,
    **kwargs: Any,
) -> SimulationResult:
    """Construct and run :class:`SingleTemperatureSimulation`."""
    return SingleTemperatureSimulation(model, metropolis, **kwargs).run()


Simulation = SingleTemperatureSimulation


def _validate_sweep_count(
    value: int,
    name: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        qualifier = f"greater than or equal to {minimum}"
        raise ValueError(f"{name} must be an integer {qualifier}")
    return int(value)