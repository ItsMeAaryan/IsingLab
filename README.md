# IsingLab

IsingLab is a Python implementation of a two-dimensional square-lattice
Ising-model Monte Carlo simulator. It is designed as a clear teaching and
experimentation project: the model, Metropolis updates, measurements,
temperature sweeps, plotting, and export layers are kept separate.

## Scientific motivation

The Ising model is a simple statistical-mechanics model of interacting
magnetic spins. Despite its local rules, it exhibits collective behavior and
a phase transition. Monte Carlo sampling makes it possible to study finite
lattices numerically and compare measured trends with exact theoretical
results.

## Hamiltonian

For spins `s_i` in `{−1, +1}`, coupling `J`, and external field `h`, IsingLab
uses

```text
H = -J sum_<i,j>(s_i s_j) - h sum_i(s_i)
```

The nearest-neighbor sum counts each periodic-lattice bond once. The lattice
has `N = L²` spins.

## Metropolis algorithm

One proposal selects a random lattice site and considers flipping its spin.
The local energy difference is

```text
ΔE = 2 J s_i sum_neighbors(s_j) + 2 h s_i
```

- If `ΔE <= 0`, the flip is accepted.
- If `ΔE > 0`, it is accepted with probability
  `exp(-ΔE / (k_B T))`.

One Monte Carlo sweep contains `L²` random-site proposals. The implementation
uses the local expression above rather than recomputing the full energy for
each proposal.

## Thermodynamic observables

Measurements store total energy `H` and signed total magnetization `M`.
Reported per-spin means are divided by `N`:

```text
mean energy per spin       = <H> / N
mean magnetization per spin = <M> / N
mean absolute magnetization per spin = <|M|> / N
```

Fluctuation observables use total quantities before normalization:

```text
C_V = (<H²> - <H>²) / (N k_B T²)
χ   = (<M²> - <M>²) / (N k_B T)
```

Susceptibility uses signed `M`; absolute magnetization is a separate
reported statistic.

## Architecture

```text
model.py          square lattice, energy, magnetization, local ΔE
metropolis.py     single-spin Metropolis proposals and sweeps
observables.py    measurements, means, fluctuation formulas
simulation.py     equilibration and single-temperature workflows
experiments.py    independent seeded temperature sweeps
visualization.py  Matplotlib plots
io.py             CSV, JSON, and NumPy exports
examples/         beginner-sized end-to-end scripts
```

## Installation

Create or activate a Python 3.10+ environment, then install the project and
its test dependencies according to your environment. The runtime requires
NumPy and Matplotlib; tests use pytest.

From the project root:

```powershell
python -m pip install -e .
```

## Usage examples

Run one simulation and save a spin heatmap plus JSON results:

```powershell
python examples/single_temperature.py
```

Run a short temperature sweep around the critical region:

```powershell
python examples/temperature_sweep.py
```

Generated files are written below `examples/output/` and are ignored by Git.
The scripts intentionally use modest lattice sizes and sweep counts so they
are suitable for a beginner's computer.

## Example outputs

The single-temperature example writes:

- `spin_configuration.png`
- `results.json`

The temperature-sweep example writes:

- `temperature_sweep.csv`
- `configuration.json`
- `energy.png`
- `magnetization.png`
- `specific_heat.png`
- `susceptibility.png`

## Reproducibility and random seeds

Pass an integer `seed` to model, simulation, or experiment constructors.
Temperature sweeps derive independent child seeds from the base seed, so
repeated runs with the same inputs reproduce their samples and observables
without sharing mutable model state between temperatures.

## Testing

Run the complete suite from the project root:

```powershell
python -m pytest -v
```

The tests cover the Hamiltonian, periodic boundaries, local flip energies,
Metropolis acceptance, normalization formulas, equilibration behavior,
temperature sweeps, exports, and non-interactive plotting.

## Finite-size limitations

For the infinite square-lattice ferromagnet with `J=1` and `h=0`, the exact
critical temperature is

```text
Tc = 2 / ln(1 + sqrt(2)) ≈ 2.269185
```

This is a thermodynamic-limit reference, not an exact prediction for a
finite simulation. Finite-size effects shift and broaden the transition, and
critical fluctuations require larger lattices and longer, better-sampled
runs. The example plots mark this value as an infinite-lattice reference.

## Known limitations

- The current sampler uses single-spin Metropolis updates only.
- There is no autocorrelation-time analysis or uncertainty estimation.
- Temperature sweeps run sequentially.
- Results are not intended to establish publication-grade precision.
- Plotting and export APIs are intentionally lightweight.

## Future improvements

Potential next steps include statistical error bars and binning, integrated
autocorrelation analysis, cluster algorithms such as Wolff updates,
parallel temperature sweeps, richer output schemas, and performance
benchmarks validated against independent reference implementations.