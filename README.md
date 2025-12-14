# Battery Arbitrage Optimization for the IQM Hackathon 2025

This repository holds the code developed by the **queen_of_fog** team for the
[IQM Hackathon 2025](https://www.meetiqm.com/hackathon).  It focuses on
solving a day-ahead battery arbitrage problem using both classical and
quantum approaches.  The goal is to determine an optimal schedule for
charging and discharging a battery in response to a deterministic price
profile and stochastic wind generation scenarios, thereby maximising
expected revenue.  This repository contains source code as
well as Jupyter notebook tutorials explaining how to build, solve and
analyse the problem.

### Tutorials

The `tutorials/` directory contains a sequence of notebooks illustrating how to
tackle the energy-storage optimisation problem.  Each file demonstrates, in
turn, how to

1. **Load and explore the data.**  We show how to read the input CSV
   containing hourly prices and 13 equiprobable wind scenarios, compute
   summary statistics (mean wind, quantiles and price statistics), and
   visualise the price curve, wind scenarios and fan charts for both wind
   production and revenue using the helper class defined in
   `src/plot_data_utils.py`.
2. **Formulate and solve the classical MILP.**  We derive a deterministic
   mixed-integer linear programme capturing state-of-charge dynamics, power
   limits, continuity constraints, cycle budgets and block limits.  Using
   the `ClassicalMILPSolver` class from `src/classical_MILP_solver.py` we
   solve the optimisation problem with PuLP and visualise the resulting
   battery dispatch and state of charge trajectory.
3. **Encode the problem as a QUBO and apply QAOA.**  We map the MILP
   constraints into a binary quadratic optimisation (QUBO) form, build a
   variational circuit implementing the Quantum Approximate Optimisation
   Algorithm (QAOA) with Pennylane, and show how to recover a schedule from
   the measured bitstrings.  This notebook demonstrates how quantum
   computers can be applied to classical scheduling problems.
4. **Combine quantum and classical techniques.**  A hybrid notebook uses
   the QAOA output to generate a warm-start pattern for the MILP.  The
   resulting schedule achieves near-optimal revenue while significantly
   reducing the classical solver’s search space.

### Code description

The core Python modules live in the `src/` directory, which plays the
same role as the `main/` folder in the VFA-Schrodinger-like-equations
repository.  Its structure is:

```sh
src/
├── __init__.py
├── plot_data_utils.py
├── classical_MILP_solver.py
└── (future modules for QUBO/QAOA)
