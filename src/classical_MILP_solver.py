"""Classical MILP solver for the day‑ahead battery arbitrage problem.

This module defines a :class:`ClassicalMILPSolver` class that formulates and
solves a deterministic mixed–integer linear program (MILP) representing
the battery scheduling problem described in the documentation.  The model
captures battery energy storage, charging/discharging dynamics, power
limits, start counting constraints and an optional penalty on the number
of starts.  It maximises expected revenue from arbitrage against a
deterministic price profile, optionally adding a constant expected wind
revenue term.

The solver relies on the `pulp` library as an MILP modelling backend.  In
order to use this module you need to install pulp (e.g. via
``pip install pulp``).  The class accepts a pandas DataFrame containing
hourly price information and returns a schedule as a DataFrame together
with solver status and objective values.

"""

from __future__ import annotations

import pandas as pd
from typing import Dict, Iterable, Tuple, List, Optional, Sequence

try:
    import pulp
except ImportError as exc:  # pragma: no cover - handled at runtime
    # The pulp library is required for optimisation; raise a clear error on import
    raise ImportError(
        "The `pulp` package is required to use ClassicalMILPSolver. "
        "Install it via `pip install pulp`.") from exc


class ClassicalMILPSolver:
    """Solver for the deterministic battery arbitrage MILP.

    Parameters
    ----------
    Pch : float, optional
        Maximum charging power in MW (default 5.0 MW).  Because time is
        discretised hourly, this also serves as the maximum charging
        energy per hour in MWh.
    Pdis : float, optional
        Maximum discharging power in MW (default 4.0 MW).
    Emax : float, optional
        Battery energy capacity in MWh (default 16.0 MWh).
    eta_ch : float, optional
        Charging efficiency (default 0.8).
    eta_dis : float, optional
        Discharging efficiency (default 1.0).
    max_cycles : int, optional
        Maximum number of equivalent full cycles (EFC) the battery may
        discharge over the horizon.  The discharge limit is
        ``max_cycles * Emax`` (default 2 cycles).
    lambda_switch : float, optional
        Penalty per start (charging or discharging block).  A value of
        zero disables the penalty (default 0.0).
    eps : float, optional
        Optional lower bound on power when a mode is active.  If set
        greater than zero, avoids the optimiser selecting a non‑zero
        binary decision variable with zero continuous power (default 0.0).

    Notes
    -----
    The solver is designed to be agnostic of scenario‑dependent wind
    production.  The constant expected wind revenue should be computed
    externally (for example using :func:`PlotDataUtils.wind_energy_summary`)
    and passed to :meth:`solve`.
    """

    def __init__(
        self,
        charge_pattern: Optional[Sequence[int]] = None,
        *,
        Pch: float = 5.0,
        Pdis: float = 4.0,
        Emax: float = 16.0,
        eta_ch: float = 0.8,
        eta_dis: float = 1.0,
        max_cycles: int = 2,
        lambda_switch: float = 0.0,
        eps: float = 0.0,
    ) -> None:
        self.Pch = Pch
        self.Pdis = Pdis
        self.Emax = Emax
        self.eta_ch = eta_ch
        self.eta_dis = eta_dis
        self.max_cycles = max_cycles
        self.lambda_switch = lambda_switch
        self.eps = eps

        # Optional warm-start pattern: 0/1 per hour indicating charging (1) or not (0).
        # If provided, it is used to set initial values for charge[t] and discharge[t]
        # as well as the mode binaries.
        self.charge_pattern: Optional[List[int]] = None
        if charge_pattern is not None:
            self.set_initial_guess(charge_pattern)

    def set_initial_guess(self, charge_pattern: Sequence[int]) -> None:
        """Set/replace the warm-start pattern.

        Parameters
        ----------
        charge_pattern : Sequence[int]
            Sequence of 0/1 values indicating whether the battery is charging
            (1) or not (0) at each hour.
        """
        pat = [int(x) for x in charge_pattern]
        if any(x not in (0, 1) for x in pat):
            raise ValueError("charge_pattern must contain only 0/1 values")
        self.charge_pattern = pat

    def _align_charge_pattern(self, hours: List[int]) -> Optional[List[int]]:
        """Align the stored warm-start pattern to the provided hour index."""
        if self.charge_pattern is None:
            return None

        pat = self.charge_pattern
        if len(pat) == len(hours):
            return list(pat)

        # Common case: a 24-vector for hours 1..24
        if len(pat) == 24 and set(hours) == set(range(1, 25)):
            return [pat[t - 1] for t in hours]

        raise ValueError(
            f"charge_pattern length ({len(pat)}) does not match number of hours ({len(hours)}), "
            "and could not be aligned to hours 1..24."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(
        self,
        df: pd.DataFrame,
        wind_rev_exp: float = 0.0,
    ) -> Tuple[pd.DataFrame, str, float, float]:
        """Formulate and solve the battery arbitrage MILP.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with two required columns: ``hour`` (int
            indexing hours from 1 to 24 inclusive) and ``price`` (float
            day‑ahead market price in EUR/MWh).
        wind_rev_exp : float, optional
            Constant expected wind revenue.  This term is added to
            the objective but does not influence the optimisation.  If
            omitted, it defaults to zero.

        Returns
        -------
        tuple of (schedule, status, battery_profit, total_revenue)
            * ``schedule`` – DataFrame containing the optimal schedule
              (charge, discharge, state of charge and modes) per hour.
            * ``status`` – Solver status string returned by pulp.
            * ``battery_profit`` – Objective contribution from battery
              arbitrage only (excluding wind revenue and start penalties).
            * ``total_revenue`` – Full objective value including wind
              revenue and penalties.
        """
        # Validate the input DataFrame
        if not {"hour", "price"}.issubset(df.columns):
            raise ValueError("Input DataFrame must contain 'hour' and 'price' columns")
        # Extract sorted list of hours and price dictionary
        hours: List[int] = sorted(df["hour"].astype(int).tolist())
        prices: Dict[int, float] = dict(zip(df["hour"].astype(int), df["price"].astype(float)))

        # Create the MILP problem
        model = pulp.LpProblem("BatteryArbitrage_Continuous", pulp.LpMaximize)

        # Decision variables
        charge = pulp.LpVariable.dicts("charge", hours, lowBound=0)
        discharge = pulp.LpVariable.dicts("discharge", hours, lowBound=0)
        soc = pulp.LpVariable.dicts("soc", [0] + hours, lowBound=0, upBound=self.Emax)

        # Mode binaries: charge, discharge, idle
        u_ch = pulp.LpVariable.dicts("u_ch", hours, cat="Binary")
        u_dis = pulp.LpVariable.dicts("u_dis", hours, cat="Binary")
        u_id = pulp.LpVariable.dicts("u_id", hours, cat="Binary")

        # Block (start) binaries
        start_ch = pulp.LpVariable.dicts("start_ch", hours, cat="Binary")
        start_dis = pulp.LpVariable.dicts("start_dis", hours, cat="Binary")

        # Initial state of charge
        model += soc[0] == 0

        # Per‑hour constraints
        for t in hours:
            # State of charge recursion: E_t = E_{t-1} + eta_ch * charge - (1/eta_dis) * discharge
            model += soc[t] == soc[t - 1] + self.eta_ch * charge[t] - (1.0 / self.eta_dis) * discharge[t]
            # Exactly one mode: charging, discharging or idle
            model += u_ch[t] + u_dis[t] + u_id[t] == 1
            # Power limits conditioned on mode
            model += charge[t] <= self.Pch * u_ch[t]
            model += discharge[t] <= self.Pdis * u_dis[t]
            # Optional tightening to avoid fractional modes without power
            if self.eps > 0:
                model += charge[t] >= self.eps * u_ch[t]
                model += discharge[t] >= self.eps * u_dis[t]

        # End‑of‑day SOC constraint
        model += soc[hours[-1]] == 0

        # Continuity constraints: forbid immediate charge–discharge reversals
        for t_prev, t in zip(hours[:-1], hours[1:]):
            model += u_ch[t_prev] + u_dis[t] <= 1
            model += u_dis[t_prev] + u_ch[t] <= 1

        # Start counting constraints
        t0 = hours[0]
        model += start_ch[t0] == u_ch[t0]
        model += start_dis[t0] == u_dis[t0]

        for t_prev, t in zip(hours[:-1], hours[1:]):
            # Charging starts at t iff u_ch[t] = 1 and u_ch[t_prev] = 0
            model += start_ch[t] >= u_ch[t] - u_ch[t_prev]
            model += start_ch[t] <= u_ch[t]
            model += start_ch[t] <= 1 - u_ch[t_prev]
            # Discharging starts at t iff u_dis[t] = 1 and u_dis[t_prev] = 0
            model += start_dis[t] >= u_dis[t] - u_dis[t_prev]
            model += start_dis[t] <= u_dis[t]
            model += start_dis[t] <= 1 - u_dis[t_prev]

        # Limit number of charging/discharging blocks
        model += pulp.lpSum(start_ch[t] for t in hours) <= 2
        model += pulp.lpSum(start_dis[t] for t in hours) <= 2

        # Cycle budget constraint: total discharged energy <= max_cycles * Emax
        model += pulp.lpSum(discharge[t] for t in hours) <= self.max_cycles * self.Emax

        # Objective: battery arbitrage revenue + wind revenue – start penalty
        battery_revenue = pulp.lpSum(prices[t] * (discharge[t] - charge[t]) for t in hours)
        start_penalty = self.lambda_switch * pulp.lpSum(start_ch[t] + start_dis[t] for t in hours)
        model += battery_revenue + wind_rev_exp - start_penalty

        # -----------------------
        # Warm start (optional)
        # -----------------------
        # The user can pass a 0/1 vector at construction time indicating whether we
        # are charging (1) or not (0). We use it as an initial guess for:
        #   - charge[t]  (continuous)  -> guess in {0,1}
        #   - discharge[t]            -> 1 - charge_guess
        #   - mode binaries u_ch/u_dis (and u_id=0)
        # and derive consistent start_* initial values.
        aligned_pat = self._align_charge_pattern(hours)
        if aligned_pat is not None:
            try:
                soc_guess = 0.0
                soc[0].setInitialValue(0.0)
                prev_ch = 0
                prev_dis = 0

                for i, t in enumerate(hours):
                    ch_on = int(aligned_pat[i])
                    dis_on = 1 - ch_on

                    # Mode binaries (idle initialised to 0 per requested pattern)
                    u_ch[t].setInitialValue(ch_on)
                    u_dis[t].setInitialValue(dis_on)
                    u_id[t].setInitialValue(0)

                    # Continuous initial guesses (per request)
                    c0 = float(ch_on)
                    d0 = float(dis_on)  # == 1 - c0

                    # Keep the warm start numerically reasonable and avoid negative SOC.
                    # (CBC may ignore infeasible MIP starts; this improves acceptance.)
                    max_d_from_soc = (soc_guess + self.eta_ch * c0) * self.eta_dis
                    if d0 > max_d_from_soc:
                        d0 = max(0.0, max_d_from_soc)

                    c0 = min(c0, self.Pch)
                    d0 = min(d0, self.Pdis)

                    charge[t].setInitialValue(c0)
                    discharge[t].setInitialValue(d0)

                    soc_guess = soc_guess + self.eta_ch * c0 - (1.0 / self.eta_dis) * d0
                    soc_guess = max(0.0, min(self.Emax, soc_guess))
                    soc[t].setInitialValue(soc_guess)

                    # Start variables derived from mode binaries
                    if i == 0:
                        start_ch[t].setInitialValue(ch_on)
                        start_dis[t].setInitialValue(dis_on)
                    else:
                        start_ch[t].setInitialValue(1 if (ch_on == 1 and prev_ch == 0) else 0)
                        start_dis[t].setInitialValue(1 if (dis_on == 1 and prev_dis == 0) else 0)

                    prev_ch, prev_dis = ch_on, dis_on
            except Exception:
                # If the backend does not support initial values, or if anything goes
                # wrong, we simply proceed without a warm start.
                aligned_pat = None

        # Solve the model using the default CBC solver; suppress solver output.
        # If available, enable CBC warm-starting when we set initial values.
        solver_kwargs = {"msg": 0}
        if aligned_pat is not None:
            try:
                import inspect

                if "warmStart" in inspect.signature(pulp.PULP_CBC_CMD.__init__).parameters:
                    solver_kwargs["warmStart"] = True
            except Exception:
                pass

        model.solve(pulp.PULP_CBC_CMD(**solver_kwargs))
        status = pulp.LpStatus.get(model.status, str(model.status))
        battery_profit = float(pulp.value(battery_revenue)) if pulp.value(battery_revenue) is not None else float("nan")
        total_revenue = float(pulp.value(model.objective)) if pulp.value(model.objective) is not None else float("nan")

        # Build the schedule DataFrame
        schedule = pd.DataFrame({
            "hour": hours,
            "charge_MWh": [charge[t].value() for t in hours],
            "discharge_MWh": [discharge[t].value() for t in hours],
            "soc_MWh": [soc[t].value() for t in hours],
            "mode_charge": [u_ch[t].value() for t in hours],
            "mode_discharge": [u_dis[t].value() for t in hours],
            "price": [prices[t] for t in hours],
        })
        # Extra columns for debugging and block counting
        schedule["mode_idle"] = [u_id[t].value() for t in hours]
        schedule["start_charge_block"] = [start_ch[t].value() for t in hours]
        schedule["start_discharge_block"] = [start_dis[t].value() for t in hours]

        return schedule, status, battery_profit, total_revenue

    def plot_battery_schedule(
        self,
        schedule: pd.DataFrame,
        Emax: float | None = None,
        figsize: tuple[float, float] = (12, 8),
        show: bool = True,
    ):
        """Plot battery dispatch (charge/discharge) and SOC trajectory.

        Parameters
        ----------
        schedule : pandas.DataFrame
            Output schedule produced by :meth:`solve`. Must contain the columns
            ``hour``, ``charge_MWh``, ``discharge_MWh`` and ``soc_MWh``.
        Emax : float, optional
            Battery capacity for setting y-limits on the SOC plot. If omitted,
            uses the solver's configured ``self.Emax``.
        figsize : tuple, optional
            Figure size (default (12, 8)).
        show : bool, optional
            If True (default), calls ``plt.show()``.

        Returns
        -------
        (fig, (ax1, ax2))
            The matplotlib figure and axes handles.
        """
        required = {"hour", "charge_MWh", "discharge_MWh", "soc_MWh"}
        missing = required.difference(schedule.columns)
        if missing:
            raise ValueError(
                "schedule is missing required columns: " + ", ".join(sorted(missing))
            )

        # Import matplotlib lazily so the solver can be imported in headless
        # environments without plotting dependencies.
        import matplotlib.pyplot as plt

        Emax_plot = self.Emax if Emax is None else float(Emax)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Dispatch bars: charge is plotted negative, discharge positive
        ax1.bar(
            schedule["hour"],
            -schedule["charge_MWh"],
            width=0.8,
            color="#1f77b4",
            alpha=0.7,
            label="Charge (MWh, negative)",
        )
        ax1.bar(
            schedule["hour"],
            schedule["discharge_MWh"],
            width=0.8,
            color="#ff7f0e",
            alpha=0.7,
            label="Discharge (MWh)",
        )
        ax1.axhline(0, color="black", linewidth=1)
        ax1.set_ylabel("Energy per hour (MWh)")
        ax1.legend(loc="upper left")
        ax1.set_title("Battery dispatch")

        # SOC trajectory
        ax2.step(
            schedule["hour"],
            schedule["soc_MWh"],
            where="mid",
            linewidth=3,
            color="#2ca02c",
        )
        ax2.set_ylabel("State of charge (MWh)")
        ax2.set_xlabel("Hour")
        ax2.set_ylim(0, Emax_plot + 1)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("SOC trajectory")

        plt.tight_layout()
        if show:
            plt.show()

        return fig, (ax1, ax2)
