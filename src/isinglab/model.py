"""Two-dimensional square-lattice Ising model."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray


class IsingModel:
    """A square-lattice Ising model with periodic boundary conditions.

    The Hamiltonian is

    .. math:: H = -J\\sum_{\\langle i,j\\rangle}s_i s_j - h\\sum_i s_i.

    Spins are stored as a two-dimensional integer NumPy array and are
    indexed as ``spins[row, column]``.

    Periodic neighbor directions are treated as bonds/directions.  Therefore,
    on an ``L=2`` lattice the left and right directions (and likewise the
    up and down directions) refer to the same physical site but contribute
    separately, as required by the square-lattice four-neighbor convention.

    Parameters
    ----------
    L:
        Linear lattice size.  The lattice contains ``L * L`` spins.
    J:
        Nearest-neighbor coupling constant.
    h:
        External magnetic field.
    initialization:
        Either ``"random"`` or ``"ordered"``.  Random initialization draws
        each spin independently and uniformly from ``{-1, +1}``.
    seed:
        Seed for random initialization.  The seed is ignored for ordered
        initialization.
    """

    def __init__(
        self,
        L: int,
        J: float = 1.0,
        h: float = 0.0,
        initialization: str = "random",
        seed: int | None = None,
    ) -> None:
        if isinstance(L, bool) or not isinstance(L, Integral) or L < 2:
            raise ValueError("L must be an integer greater than or equal to 2")
        if not isinstance(J, Real) or not np.isfinite(J):
            raise ValueError("J must be a finite real number")
        if not isinstance(h, Real) or not np.isfinite(h):
            raise ValueError("h must be a finite real number")
        if initialization not in {"random", "ordered"}:
            raise ValueError("initialization must be either 'random' or 'ordered'")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, Integral)
        ):
            raise ValueError("seed must be an integer or None")

        self.L = int(L)
        self.J = float(J)
        self.h = float(h)
        self.rng = np.random.default_rng(seed)

        if initialization == "ordered":
            self.spins: NDArray[np.int8] = np.ones((self.L, self.L), dtype=np.int8)
        else:
            self.spins = self.rng.choice(
                np.array([-1, 1], dtype=np.int8),
                size=(self.L, self.L),
            ).astype(np.int8, copy=False)

    def energy(self) -> float:
        """Return the total energy, counting each bond exactly once."""
        horizontal_bonds = self.spins * np.roll(self.spins, -1, axis=1)
        vertical_bonds = self.spins * np.roll(self.spins, -1, axis=0)
        interaction = np.sum(horizontal_bonds, dtype=np.int64) + np.sum(
            vertical_bonds, dtype=np.int64
        )
        field = np.sum(self.spins, dtype=np.int64)
        return float(-self.J * interaction - self.h * field)

    def magnetization(self) -> int:
        """Return the total magnetization ``sum(spins)``."""
        return int(np.sum(self.spins, dtype=np.int64))

    def magnetization_per_spin(self) -> float:
        """Return the magnetization divided by the number of spins."""
        return self.magnetization() / (self.L * self.L)

    def local_energy_change(self, row: int, column: int) -> float:
        """Return the energy change caused by flipping one spin.

        The four-neighbor sum uses periodic boundaries:

        .. math:: \\Delta E = 2J s_i\\sum_{j\\in nn(i)}s_j + 2h s_i.
        """
        self._validate_coordinate(row, column)
        spin = int(self.spins[row, column])
        neighbors = (
            int(self.spins[(row - 1) % self.L, column])
            + int(self.spins[(row + 1) % self.L, column])
            + int(self.spins[row, (column - 1) % self.L])
            + int(self.spins[row, (column + 1) % self.L])
        )
        return float(2.0 * self.J * spin * neighbors + 2.0 * self.h * spin)

    def flip_spin(self, row: int, column: int) -> float:
        """Flip one spin in place and return its local energy change."""
        delta_energy = self.local_energy_change(row, column)
        self.spins[row, column] *= -1
        return delta_energy

    def flip_energy_change(self, row: int, column: int) -> float:
        """Return the energy change for flipping one spin.

        This is an explicit alias for :meth:`local_energy_change`, provided
        for Monte Carlo algorithms that describe a proposed flip directly.
        """
        return self.local_energy_change(row, column)

    def _validate_coordinate(self, row: int, column: int) -> None:
        if (
            isinstance(row, bool)
            or isinstance(column, bool)
            or not isinstance(row, Integral)
            or not isinstance(column, Integral)
            or not (0 <= row < self.L)
            or not (0 <= column < self.L)
        ):
            raise IndexError(
                f"spin coordinate ({row}, {column}) is outside a {self.L}x{self.L} lattice"
            )