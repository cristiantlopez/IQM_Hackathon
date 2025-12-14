# Battery Arbitrage Optimization for the IQM Hackathon 2025

This repository holds the code developed by the **queen_of_fog** team for the
[IQM Hackathon 2025 Endesa Challenge](https://github.com/iqm-finland/quantum-hack-2025-madrid/tree/main/endesa). It tackles a
day-ahead battery arbitrage and grid-tracking problem that combines classical
optimisation (MILP) with a quantum-inspired warm start (QUBO formulation+ QAOA solution). The
project works with a deterministic 24-hour price profile and 13 equiprobable
wind scenarios stored in `data/input_data.csv`, and packages the workflow into
reusable Python modules plus tutorial notebooks.

### Tutorials

The `tutorials/` directory walks through the full workflow:

1. **1_Data_analysis.ipynb** - load `data/input_data.csv`, explore price and 13 wind scenarios with `PlotDataUtils`, compute wind and revenue statistics, and visualise correlations between wind, revenue, and price.
2. **2_QAOA_Hardware.ipynb** - build the battery-dispatch QUBO, run QAOA (both simulators and real IQM hardware) with `QAOAGuessSolver`, and export the most probable bitstring/discharge pattern.
3. **3_Hybrid_quantum+classical_MILP.ipynb** - decode the QAOA bitstring into a warm start for `ClassicalMILPSolver`, solve the deterministic arbitrage MILP with the PuLP package, and plot the resulting dispatch and state-of-charge trajectory.
4. **4_Introducing_weather_forecasting.ipynb** - extend the MILP to multiple wind scenarios using `ClassicalMILPSolverWithWeatherData`, build a price-elastic demand target, evaluate tracking penalties, and compare or aggregate schedules across scenarios.

### Code description

Core modules live in `src/` and mirror the notebook workflow:

```text
.
|-- data/
|   |-- input_data.csv               # Hourly price + 13 wind scenarios
|   |-- qubo_matrix_symmetric.csv    # Pre-built QUBO matrix (symmetric form)
|   |-- qubo_matrix_upper.csv        # Upper-triangular QUBO matrix
|   `-- qubo_solution.csv            # Stored QAOA bitstring/discharge guess
|-- src/
|   |-- __init__.py
|   |-- plot_data_utils.py           # Load data, compute wind/revenue stats, and plot curves/fan charts
|   |-- classical_MILP_solver.py     # Deterministic battery arbitrage MILP with warm starts and plotting
|   |-- classical_MILP_solver_with_weather_data.py  # Scenario-based arbitrage/tracking MILP and visualisations
|   `-- qaoa_guess_solver.py         # Build QUBO matrix and solve via QAOA to seed the MILP
|-- tutorials/                       # Main notebooks (see above)
`-- test/                            # Scratch notebooks and solver outputs (MPS/solution files)
```

Dependencies: Python 3.11+ with `pandas`, `numpy`, `matplotlib`, `seaborn`,
`pulp`, and (for QAOA) `qiskit`, `qiskit-optimization`, `qiskit-aer`,
`qiskit-ibm-runtime`, and `scipy`. Activate the existing `.venv` or install
packages in a new environment before running notebooks.

Example usage of the classical solver:

```python
import pandas as pd
from src.classical_MILP_solver import ClassicalMILPSolver

df = pd.read_csv("data/input_data.csv")[["hour", "price"]]
solver = ClassicalMILPSolver(lambda_switch=5.0)
schedule, status, battery_profit, total = solver.solve(df)
print(status, battery_profit)
```
