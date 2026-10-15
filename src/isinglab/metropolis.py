"""Metropolis Monte Carlo updates for the Ising model."""

from __future__ import annotations

from numbers import Real

import numpy as np

from .model import IsingModel


class Metropolis:
    """Run single-spin Metropolis updates for an :class:`IsingModel`.

    A Monte Carlo sweep consists of ``L**2`` independently selected site
    proposals.  Each proposal uses the model's random generator, so a model
    initialized with a fixed seed produces a reproducible simulation.
    """

    def __init__(
        self,
        model: IsingModel,
        temperature: float,
        k_B: float = 1.0,
    ) -> None:
        if not isinstance(model, IsingModel):
            raise TypeError("model must be an IsingModel instance")
        if not isinstance(temperature, Real) or not np.isfinite(temperature):
            raise ValueError("temperature must be a finite positive number")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not isinstance(k_B, Real) or not np.isfinite(k_B):
            raise ValueError("k_B must be a finite positive number")
        if k_B <= 0:
            raise ValueError("k_B must be positive")

        self.model = model
        self.temperature = float(temperature)
        self.k_B = float(k_B)
        self.attempted_flips = 0
        self.accepted_flips = 0

    def attempt_flip(self) -> bool:
        """Propose and process one randomly selected spin flip.

        The energy difference is calculated locally.  Accepted flips update
        the model in place; rejected flips leave it unchanged.
        """
        row, column = self.model.rng.integers(0, self.model.L, size=2)
        row = int(row)
        column = int(column)
        delta_energy = self.model.flip_energy_change(row, column)

        self.attempted_flips += 1
        accepted = delta_energy <= 0 or (
            self.model.rng.random()
            < np.exp(-delta_energy / (self.k_B * self.temperature))
        )
        if accepted:
            self.model.flip_spin(row, column)
            self.accepted_flips += 1
        return bool(accepted)

    def sweep(self) -> int:
        """Perform one sweep and return the number of accepted flips."""
        accepted_before = self.accepted_flips
        for _ in range(self.model.L * self.model.L):
            self.attempt_flip()
        return self.accepted_flips - accepted_before

    @property
    def acceptance_rate(self) -> float:
        """Return accepted proposals divided by attempted proposals."""
        if self.attempted_flips == 0:
            return 0.0
        return self.accepted_flips / self.attempted_flips