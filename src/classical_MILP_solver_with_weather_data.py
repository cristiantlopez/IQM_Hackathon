"""
Classical MILP solver extended to incorporate day‑ahead wind scenario data.

This module defines a class that wraps the complete optimisation workflow for a
grid‐connected battery in the presence of uncertain wind generation.  It is
modelled on the style of ``ClassicalMILPSolver`` but accepts a pandas
``DataFrame`` containing price and wind scenario columns upon construction.  In
addition to solving a battery arbitrage/tracking problem under multiple
scenarios, the class provides helper methods to compute demand curves, track
targets, evaluate correlations and visualise both individual and aggregate
dispatch/soc schedules.  The goal is to encapsulate the lengthy code blocks
from the notebook into reusable functions following the repository’s
object‑oriented pattern【841953650875668†L3-L9】.


"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import pulp  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PuLP is required for ClassicalMILPSolverWithWeatherData. Please install it via `pip install pulp`."
    ) from exc

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ScenarioResult:
    """Container for storing per‑scenario optimisation results.

    Parameters
    ----------
    schedule : pandas.DataFrame
        Table containing hour, price, wind generation, baseline signal, charge,
        discharge and state of charge along with mode flags and penalties.
    battery_revenue : float
        ∑\_t p\_t (discharge\_t − charge\_t).
    tracking_penalty : float
        γ × ∑\_t |(discharge − charge) − p\_sig[t]|.
    start_penalty : float
        λ × number of charging/discharging block starts.
    wind_revenue : float
        Per‑scenario wind revenue (constant w.r.t. battery decisions).
    solver_status : str
        PuLP solver status string.
    """

    schedule: pd.DataFrame
    battery_revenue: float
    tracking_penalty: float
    start_penalty: float
    wind_revenue: float
    solver_status: str


class ClassicalMILPSolverWithWeatherData:
    """Solve battery arbitrage/tracking MILPs across multiple wind scenarios.

    This class extends the original ``ClassicalMILPSolver`` by accepting a
    ``DataFrame`` containing hour, price and multiple wind scenario columns
    (named ``scenario_1``, ``scenario_2``, …).  It constructs a demand curve
    proportional to price^(-ϵ) such that the total demand equals the average
    total wind generation, and solves a series of MILP problems—one per
    scenario—where the battery attempts to track the net supply signal and
    optionally arbitrage energy prices.  Results are returned as a list of
    :class:`ScenarioResult` objects, and several plotting routines are provided
    to visualise individual schedules, compare selected scenarios and
    summarise aggregated dispatch/soc statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing columns ``hour``, ``price`` and
        ``scenario_1``..``scenario_n``.  ``hour`` may be any sortable
        identifiers but is converted to integers for modelling.
    Pch : float, default 5.0
        Maximum charging power (MW).  Energy per hour is limited by this
        quantity.
    Pdis : float, default 4.0
        Maximum discharging power (MW).
    Emax : float, default 16.0
        Energy capacity of the battery (MWh).
    eta_ch : float, default 0.8
        Charging efficiency.
    eta_dis : float, default 1.0
        Discharging efficiency.
    max_cycles : int, default 2
        Equivalent full cycle budget.  The total discharged energy is
        constrained to be ≤ ``max_cycles * Emax``.
    charge_pattern : Optional[Sequence[int]], default None
        Optional initial guess for charging schedule.  If provided, it should
        be a sequence of length equal to the number of hours with elements 0 or
        1 indicating whether the solver should try to start with charging (1)
        or not (0).  A matching discharging guess of ``1 - charge_pattern``
        will be used.  The binary mode variables and start counts are
        initialised accordingly.  If the pattern violates continuity
        constraints, CBC may ignore the warm start.

    Notes
    -----
    The optimisation models are solved with PuLP’s default CBC solver.  Warm
    starts are passed using the ``pulp.LpVariable.setInitialValue`` method and
    enabling ``warmStart=True`` on the solver command.  If CBC encounters an
    infeasible warm start due to conflicting continuity or block constraints it
    may silently drop the initialisation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        Pch: float = 5.0,
        Pdis: float = 4.0,
        Emax: float = 16.0,
        eta_ch: float = 0.8,
        eta_dis: float = 1.0,
        max_cycles: int = 2,
        charge_pattern: Optional[Sequence[int]] = None,
    ) -> None:
        # Copy and validate input data
        if not isinstance(df, pd.DataFrame):  # pragma: no cover
            raise TypeError("df must be a pandas DataFrame")
        self.df = df.copy()
        # Ensure hour is integer for consistent indexing
        if "hour" not in self.df.columns or "price" not in self.df.columns:
            raise ValueError("df must contain 'hour' and 'price' columns")
        self.df["hour"] = self.df["hour"].astype(int)
        self.scenario_cols: List[str] = [c for c in self.df.columns if c.startswith("scenario_")]
        if not self.scenario_cols:
            raise ValueError("df must contain at least one 'scenario_*' column")
        # Model horizon hours sorted ascending
        self.hours: List[int] = sorted(self.df["hour"].unique().astype(int).tolist())
        # Battery parameters
        self.Pch = float(Pch)
        self.Pdis = float(Pdis)
        self.Emax = float(Emax)
        self.eta_ch = float(eta_ch)
        self.eta_dis = float(eta_dis)
        self.max_cycles = int(max_cycles)
        # Warm‑start pattern if provided
        if charge_pattern is not None:
            if len(charge_pattern) != len(self.hours):
                raise ValueError(
                    f"charge_pattern must have length {len(self.hours)}, got {len(charge_pattern)}"
                )
            # ensure values are 0 or 1
            for x in charge_pattern:
                if x not in (0, 1):
                    raise ValueError("charge_pattern elements must be 0 or 1")
            self.charge_pattern: Optional[List[int]] = list(map(int, charge_pattern))
        else:
            self.charge_pattern = None

    # ------------------------------------------------------------------
    # Demand and wind processing
    # ------------------------------------------------------------------
    def compute_winds(self) -> np.ndarray:
        """Return a 2‑D numpy array of wind generation by scenario.

        The array has shape (n_hours, n_scenarios) such that ``winds[t, s]`` is
        the wind generation at hour index ``t`` (0‑based, corresponding to
        ``self.hours[t]``) under scenario ``s`` (0‑based index for
        ``self.scenario_cols``).
        """
        n_hours = len(self.hours)
        n_scenarios = len(self.scenario_cols)
        winds = np.zeros((n_hours, n_scenarios), dtype=float)
        for i, col in enumerate(self.scenario_cols):
            winds[:, i] = self.df[col].astype(float).to_numpy()
        return winds

    def compute_demand(self, epsilon: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the price‑elastic demand curve and return winds array.

        Demand is proportional to ``price**(-epsilon)`` scaled such that the
        aggregate demand equals the average total wind generation across all
        scenarios.  This method also returns the winds array for convenience.

        Parameters
        ----------
        epsilon : float, default 0.25
            Elasticity exponent.  A value in (0, 1) yields a decreasing demand
            with respect to price.

        Returns
        -------
        winds : numpy.ndarray
            Two‑dimensional array of shape (n_hours, n_scenarios) of wind
            generation.
        demand : numpy.ndarray
            One‑dimensional array of length ``n_hours`` representing the
            demand at each hour, aligned with ``self.hours``.
        """
        winds = self.compute_winds()
        # average total wind across scenarios for each hour, then sum over hours
        avg_total_wind = float(np.sum(np.mean(winds, axis=1)))
        price = self.df.set_index("hour").reindex(self.hours)["price"].astype(float).to_numpy()
        # Avoid division by zero in price**(-epsilon)
        with np.errstate(divide="ignore", invalid="ignore"):
            price_power = np.where(price > 0.0, price ** (-epsilon), 0.0)
        total_price_power = np.sum(price_power)
        if total_price_power == 0.0:
            # Fallback: uniform demand if prices are zero or very small
            demand = np.full_like(price, avg_total_wind / len(price), dtype=float)
        else:
            A = avg_total_wind / total_price_power
            demand = A * price_power
        return winds, demand

    # ------------------------------------------------------------------
    # Target correlation and plotting
    # ------------------------------------------------------------------
    def compute_target_stats(
        self,
        targets: np.ndarray,
        quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float, pd.Series]:
        """Compute per‑scenario target statistics and correlations.

        Given a matrix of ``targets`` with shape (n_hours, n_scenarios) this
        method constructs a ``DataFrame`` indexed by ``self.hours`` with
        columns corresponding to each scenario, as well as a ``DataFrame`` of
        row‑wise quantiles.  It then computes the correlation between the mean
        target and the price series, and the distribution of correlations
        between each scenario and price.

        Parameters
        ----------
        targets : numpy.ndarray
            Two‑dimensional array of shape (n_hours, n_scenarios) containing
            values to treat as per‑scenario targets.
        quantiles : sequence of float, optional
            Probabilities at which to compute quantiles of the targets along
            the scenario axis.

        Returns
        -------
        target_df : pandas.DataFrame
            DataFrame with index ``self.hours`` and one column per scenario.
        target_quant_df : pandas.DataFrame
            DataFrame with index ``self.hours`` and one column per quantile.
        corr_mean : float
            Correlation between the mean target (across scenarios) and price.
        corr_by_scenario : pandas.Series
            Series of correlations between each scenario and the price series.
        """
        n_hours, n_scenarios = targets.shape
        if n_hours != len(self.hours) or n_scenarios != len(self.scenario_cols):
            raise ValueError(
                f"targets must have shape ({len(self.hours)}, {len(self.scenario_cols)})"
            )
        target_df = pd.DataFrame(targets, index=self.hours, columns=self.scenario_cols)
        # compute mean and quantiles along scenario axis
        target_mean = target_df.mean(axis=1)
        target_quant_df = pd.DataFrame({q: target_df.quantile(q, axis=1) for q in quantiles})
        # compute correlations with price
        price_series = self.df.set_index("hour").reindex(self.hours)["price"].astype(float)
        corr_mean = float(target_mean.corr(price_series))
        corr_by_scenario = target_df.apply(lambda s: s.corr(price_series))
        return target_df, target_quant_df, corr_mean, corr_by_scenario

    def plot_target_vs_price(
        self,
        target_df: pd.DataFrame,
        target_quant_df: pd.DataFrame,
        corr_mean: float,
        corr_by_scenario: pd.Series,
        *,
        palette: Optional[Sequence] = None,
        figsize: Tuple[float, float] = (12, 5),
        show: bool = True,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """Visualise targets and their quantiles against price.

        Produces two overlay plots: one showing each scenario target curve and
        their mean with price overlaid on a secondary axis; the second showing
        selected quantile bands of the targets with price on a twin axis.  The
        computed correlations are printed to the console.  This method
        intentionally mirrors the plotting style in the original notebook.

        Parameters
        ----------
        target_df : pandas.DataFrame
            Table of per‑scenario targets indexed by hour.
        target_quant_df : pandas.DataFrame
            Quantile summary of the targets (output of
            :meth:`compute_target_stats`).
        corr_mean : float
            Correlation of the mean target with price (for printing).
        corr_by_scenario : pandas.Series
            Series of correlations of each scenario with price (for printing).
        palette : sequence, optional
            Colour palette to use for the individual scenario lines.
        figsize : tuple of float, optional
            Overall figure size.
        show : bool, default True
            If True, ``plt.show()`` is called at the end.

        Returns
        -------
        (fig1, (ax1a, ax1b)), (fig2, (ax2a, ax2b)) : tuple
            Handles to the two figures and their axes.
        """
        # Print correlations
        mean_corr_str = f"Corr(price, mean target): {corr_mean:.3f}"
        across = corr_by_scenario
        scatter = (
            f"Corr(price, target) across scenarios: mean={across.mean():.3f}, "
            f"min={across.min():.3f}, max={across.max():.3f}"
        )
        print(mean_corr_str)
        print(scatter)
        # Palette for scenario lines
        if palette is None:
            palette = sns.color_palette("crest", n_colors=target_df.shape[1])
        # Overlay plot: scenarios + price
        fig1, ax1 = plt.subplots(figsize=figsize)
        ax_price = ax1.twinx()
        for colour, col in zip(palette, target_df.columns):
            ax1.plot(target_df.index, target_df[col], color=colour, alpha=0.35, linewidth=2)
        # mean target
        ax1.plot(target_df.index, target_df.mean(axis=1), color="black", linewidth=3, label="Scenario mean target")
        ax1.set_xlabel("Hour of day")
        ax1.set_ylabel("Target (MWh)")
        # Price overlay
        price_series = self.df.set_index("hour").reindex(self.hours)["price"].astype(float)
        ax_price.plot(price_series.index, price_series.values, color="crimson", linewidth=2.5, marker="o", label="Price")
        ax_price.set_ylabel("Market price (EUR/MWh)")
        ax1.set_title("Hourly target levels vs price")
        ax1.legend(loc="upper left")
        ax_price.legend(loc="upper right")
        fig1.tight_layout()
        # Overlay plot: quantiles + price
        fig2, ax2 = plt.subplots(figsize=figsize)
        ax2_price = ax2.twinx()
        # quantile curves
        q50 = target_quant_df.get(0.5, None)
        if q50 is not None:
            ax2.plot(target_quant_df.index, q50, color="black", linewidth=3, label="Median target")
        # fill between quantiles if available
        q25 = target_quant_df.get(0.25, None)
        q75 = target_quant_df.get(0.75, None)
        q10 = target_quant_df.get(0.1, None)
        q90 = target_quant_df.get(0.9, None)
        if q25 is not None and q75 is not None:
            ax2.fill_between(target_quant_df.index, q25, q75, color="#1f77b4", alpha=0.25, label="25–75% band")
        if q10 is not None and q90 is not None:
            ax2.fill_between(target_quant_df.index, q10, q90, color="#1f77b4", alpha=0.12, label="10–90% band")
        ax2.set_xlabel("Hour of day")
        ax2.set_ylabel("Target (MWh)")
        # Price overlay
        ax2_price.plot(price_series.index, price_series.values, color="crimson", linewidth=2.5, marker="o", label="Price")
        ax2_price.set_ylabel("Market price (EUR/MWh)")
        ax2.set_title("Hourly target quantiles vs price")
        ax2.legend(loc="upper left")
        ax2_price.legend(loc="upper right")
        fig2.tight_layout()
        if show:
            plt.show()
        return (fig1, (ax1, ax_price)), (fig2, (ax2, ax2_price))

    # ------------------------------------------------------------------
    # MILP solving across scenarios
    # ------------------------------------------------------------------
    def solve(
        self,
        *,
        gamma_track: float = 1.0,
        lambda_switch: float = 0.0,
        eps: float = 0.0,
    ) -> Tuple[List[ScenarioResult], dict]:
        """Solve one MILP per wind scenario and return schedules and summary.

        Parameters
        ----------
        gamma_track : float, default 1.0
            Weight applied to the absolute tracking deviation in the objective.
        lambda_switch : float, default 0.0
            Weight applied to the number of charging/discharging block starts.
        eps : float, default 0.0
            Optional tightening parameter ensuring that if a mode is active
            (charge or discharge), the corresponding power is at least ``eps``.

        Returns
        -------
        schedules : list of ScenarioResult
            A list containing an optimisation result for each scenario.
        summary : dict
            Aggregate statistics including expected wind revenue, range of
            scenario wind revenues and average battery revenue.
        """
        winds, demand = self.compute_demand()
        # Align price series to hours
        df_h = self.df.copy()
        df_h["hour"] = df_h["hour"].astype(int)
        price_by_hour = df_h.set_index("hour").reindex(self.hours)["price"].astype(float).to_dict()
        n_hours = len(self.hours)
        # Pre‑compute per scenario wind revenue for summary
        wind_rev_by_s = (self.df[self.scenario_cols].mul(self.df["price"], axis=0)).sum(axis=0)
        wind_rev_exp = float(wind_rev_by_s.mean())
        wind_rev_min = float(wind_rev_by_s.min())
        wind_rev_max = float(wind_rev_by_s.max())
        schedules: List[ScenarioResult] = []
        # Iterate over each scenario
        for s, col in enumerate(self.scenario_cols):
            # wind generation series for scenario s
            wind_gen = winds[:, s]  # length n_hours
            # baseline signal p_sig[t] = demand[t-1] - wind[t,s]
            p_sig = {t: float(demand[idx] - wind_gen[idx]) for idx, t in enumerate(self.hours)}
            model = pulp.LpProblem(f"Battery_ArbPlusTracking_S{s+1}", pulp.LpMaximize)
            # Decision variables: energy in/out and state of charge
            charge = pulp.LpVariable.dicts("charge", self.hours, lowBound=0)
            discharge = pulp.LpVariable.dicts("discharge", self.hours, lowBound=0)
            soc = pulp.LpVariable.dicts("soc", [0] + self.hours, lowBound=0, upBound=self.Emax)
            # 3‑state mode binaries
            y_ch = pulp.LpVariable.dicts("y_ch", self.hours, cat="Binary")
            y_dis = pulp.LpVariable.dicts("y_dis", self.hours, cat="Binary")
            y_id = pulp.LpVariable.dicts("y_id", self.hours, cat="Binary")
            # Start/block binaries
            u_ch = pulp.LpVariable.dicts("u_ch", self.hours, cat="Binary")
            u_dis = pulp.LpVariable.dicts("u_dis", self.hours, cat="Binary")
            # Tracking variables
            dev = pulp.LpVariable.dicts("dev", self.hours, lowBound=None)  # free variable
            abs_dev = pulp.LpVariable.dicts("abs_dev", self.hours, lowBound=0)
            # Initial SOC
            model += soc[0] == 0
            # Warm start: if charge_pattern provided, initialise variables
            warm_start = self.charge_pattern is not None
            if warm_start:
                # compute warm starts for binary modes and power variables
                patt = self.charge_pattern
                for idx, t in enumerate(self.hours):
                    c_val = float(patt[idx])
                    d_val = float(1 - patt[idx])
                    # initial power guesses (within bounds)
                    charge[t].setInitialValue(c_val)
                    discharge[t].setInitialValue(d_val)
                    y_ch[t].setInitialValue(c_val)
                    y_dis[t].setInitialValue(d_val)
                    y_id[t].setInitialValue(0)
                    # start flags from transitions
                    if idx == 0:
                        u_ch[t].setInitialValue(c_val)
                        u_dis[t].setInitialValue(d_val)
                    else:
                        # start if 0->1 transition
                        prev_c = float(patt[idx - 1])
                        prev_d = float(1 - patt[idx - 1])
                        u_ch[t].setInitialValue(max(0.0, c_val - prev_c))
                        u_dis[t].setInitialValue(max(0.0, d_val - prev_d))
            # Per‑hour constraints
            for idx, t in enumerate(self.hours):
                # SOC recursion: E_t = E_{t-1} + η_ch * c_t - (1/η_dis) * d_t
                model += soc[t] == soc[self.hours[idx - 1] if idx > 0 else 0] + self.eta_ch * charge[t] - (1.0 / self.eta_dis) * discharge[t]
                # 3‑state operation
                model += y_ch[t] + y_dis[t] + y_id[t] == 1
                # Power limits
                model += charge[t] <= self.Pch * y_ch[t]
                model += discharge[t] <= self.Pdis * y_dis[t]
                # Optional tightening
                if eps > 0:
                    model += charge[t] >= eps * y_ch[t]
                    model += discharge[t] >= eps * y_dis[t]
                # Tracking deviation: dev[t] = (discharge - charge) - p_sig[t]
                model += dev[t] == (discharge[t] - charge[t]) - p_sig[t]
                # absolute value linearisation
                model += abs_dev[t] >= dev[t]
                model += abs_dev[t] >= -dev[t]
            # Terminal SOC
            model += soc[self.hours[-1]] == 0
            # Continuous operation: no immediate charge<->discharge reversals
            for t_prev, t_cur in zip(self.hours[:-1], self.hours[1:]):
                model += y_ch[t_prev] + y_dis[t_cur] <= 1
                model += y_dis[t_prev] + y_ch[t_cur] <= 1
            # Start definition constraints
            t0 = self.hours[0]
            model += u_ch[t0] == y_ch[t0]
            model += u_dis[t0] == y_dis[t0]
            for t_prev, t_cur in zip(self.hours[:-1], self.hours[1:]):
                # Charging start if y_ch goes 0->1
                model += u_ch[t_cur] >= y_ch[t_cur] - y_ch[t_prev]
                model += u_ch[t_cur] <= y_ch[t_cur]
                model += u_ch[t_cur] <= 1 - y_ch[t_prev]
                # Discharging start if y_dis goes 0->1
                model += u_dis[t_cur] >= y_dis[t_cur] - y_dis[t_prev]
                model += u_dis[t_cur] <= y_dis[t_cur]
                model += u_dis[t_cur] <= 1 - y_dis[t_prev]
            # Limit number of blocks
            model += pulp.lpSum(u_ch[t] for t in self.hours) <= 2
            model += pulp.lpSum(u_dis[t] for t in self.hours) <= 2
            # Cycle proxy: total discharged energy <= max_cycles * Emax
            model += pulp.lpSum(discharge[t] for t in self.hours) <= self.max_cycles * self.Emax
            # Objective components
            battery_revenue = pulp.lpSum(price_by_hour[t] * (discharge[t] - charge[t]) for t in self.hours)
            # track_penalty = γ × Σ |dev|
            track_penalty = gamma_track * pulp.lpSum(abs_dev[t] for t in self.hours)
            # start_penalty = λ × Σ (u_ch + u_dis)
            start_penalty = lambda_switch * pulp.lpSum(u_ch[t] + u_dis[t] for t in self.hours)
            # Scenario wind revenue (constant w.r.t. battery decisions)
            wind_revenue_s = sum(price_by_hour[t] * float(wind_gen[idx]) for idx, t in enumerate(self.hours))
            # Objective: maximise battery revenue minus penalties plus constant wind term
            #model += battery_revenue - track_penalty - start_penalty + wind_revenue_s
            # Objective: maximise the SPT problem
            model += - track_penalty - start_penalty
            # Solve with warm start flag if provided
            solver = pulp.PULP_CBC_CMD(msg=0, warmStart=warm_start)
            model.solve(solver)
            status_str = pulp.LpStatus[model.status]
            batt_rev = float(pulp.value(battery_revenue))
            trk_pen_val = float(pulp.value(track_penalty))
            sw_pen_val = float(pulp.value(start_penalty))
            # Build schedule DataFrame
            schedule = pd.DataFrame({
                "hour": self.hours,
                "price": [price_by_hour[t] for t in self.hours],
                "wind_MWh": [float(wind_gen[idx]) for idx in range(n_hours)],
                "p_sig": [p_sig[t] for t in self.hours],
                "charge_MWh": [pulp.value(charge[t]) for t in self.hours],
                "discharge_MWh": [pulp.value(discharge[t]) for t in self.hours],
                "soc_MWh": [pulp.value(soc[t]) for t in self.hours],
                "mode_charge": [pulp.value(y_ch[t]) for t in self.hours],
                "mode_discharge": [pulp.value(y_dis[t]) for t in self.hours],
                "mode_idle": [pulp.value(y_id[t]) for t in self.hours],
                "start_charge_block": [pulp.value(u_ch[t]) for t in self.hours],
                "start_discharge_block": [pulp.value(u_dis[t]) for t in self.hours],
                "dev": [pulp.value(dev[t]) for t in self.hours],
                "abs_dev": [pulp.value(abs_dev[t]) for t in self.hours],
            })
            result = ScenarioResult(
                schedule=schedule,
                battery_revenue=batt_rev,
                tracking_penalty=trk_pen_val,
                start_penalty=sw_pen_val,
                wind_revenue=float(wind_revenue_s),
                solver_status=status_str,
            )
            schedules.append(result)
        # Summary dictionary
        summary = {
            "expected_wind_revenue": wind_rev_exp,
            "wind_revenue_min": wind_rev_min,
            "wind_revenue_max": wind_rev_max,
            "mean_battery_revenue": np.mean([res.battery_revenue for res in schedules]),
            "mean_tracking_penalty": np.mean([res.tracking_penalty for res in schedules]),
            "mean_start_penalty": np.mean([res.start_penalty for res in schedules]),
        }
        return schedules, summary

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def plot_battery_schedule(
        self,
        schedule: pd.DataFrame,
        *,
        Emax: Optional[float] = None,
        figsize: Tuple[float, float] = (12, 8),
        show: bool = True,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """Plot battery dispatch (charge/discharge) and SOC for a single schedule.

        Parameters
        ----------
        schedule : pandas.DataFrame
            Output schedule as returned by :meth:`solve`.  Must contain
            ``hour``, ``charge_MWh``, ``discharge_MWh`` and ``soc_MWh``.
        Emax : float, optional
            Maximum battery capacity used to bound the SOC plot.  Defaults to
            ``self.Emax``.
        figsize : tuple, optional
            Figure size.  Defaults to (12, 8).
        show : bool, optional
            If True (default), calls ``plt.show()``.

        Returns
        -------
        (fig, (ax1, ax2)) : tuple
            Matplotlib figure and axes objects.
        """
        required = {"hour", "charge_MWh", "discharge_MWh", "soc_MWh"}
        missing = required.difference(schedule.columns)
        if missing:
            raise ValueError(f"schedule is missing required columns: {', '.join(sorted(missing))}")
        Emax_plot = float(self.Emax) if Emax is None else float(Emax)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        # dispatch bars: charge negative, discharge positive
        ax1.bar(
            schedule["hour"],
            -np.array(schedule["charge_MWh"], dtype=float),
            width=0.8,
            color="#1f77b4",
            alpha=0.7,
            label="Charge (MWh, negative)",
        )
        ax1.bar(
            schedule["hour"],
            np.array(schedule["discharge_MWh"], dtype=float),
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
        ax2.step(schedule["hour"], schedule["soc_MWh"], where="mid", linewidth=3, color="#2ca02c")
        ax2.set_ylabel("State of charge (MWh)")
        ax2.set_xlabel("Hour")
        ax2.set_ylim(0, Emax_plot + 1)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("SOC trajectory")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, (ax1, ax2)

    def plot_dispatch_comparison(
        self,
        schedules: Sequence[ScenarioResult],
        *,
        indices: Sequence[int],
        figsize: Tuple[float, float] = (18, 8),
        Emax: Optional[float] = None,
        show: bool = True,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot dispatch and SOC for multiple scenario schedules in a grid.

        Parameters
        ----------
        schedules : sequence of ScenarioResult
            Results returned by :meth:`solve`.
        indices : sequence of int
            Scenario indices (0‑based) to include in the comparison.  There will
            be one column per index in the resulting plot.
        figsize : tuple, optional
            Figure size.  Defaults to (18, 8).
        Emax : float, optional
            Battery capacity for setting the y‑limits on the SOC plots.
        show : bool, optional
            If True (default), calls ``plt.show()``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The resulting figure.
        axes : numpy.ndarray
            Array of axes objects arranged in 2 rows and ``len(indices)`` columns.
        """
        if not indices:
            raise ValueError("indices must be a non‑empty sequence of scenario indices")
        n_cols = len(indices)
        fig, axes = plt.subplots(2, n_cols, figsize=figsize, sharex=True)
        # flatten axes for consistent indexing when n_cols = 1
        axes = np.array(axes).reshape(2, n_cols)
        charge_color = "#1f77b4"
        discharge_color = "#ff7f0e"
        soc_color = "#2ca02c"
        Emax_plot = float(self.Emax) if Emax is None else float(Emax)
        for col_idx, idx in enumerate(indices):
            if idx < 0 or idx >= len(schedules):
                raise IndexError(f"scenario index {idx} out of range")
            sched_df = schedules[idx].schedule
            hours = sched_df["hour"]
            # Dispatch bars
            ax_top = axes[0, col_idx]
            ax_top.bar(hours, -np.array(sched_df["charge_MWh"], dtype=float), width=0.8, color=charge_color, alpha=0.7, label="Charge (MWh, negative)")
            ax_top.bar(hours, np.array(sched_df["discharge_MWh"], dtype=float), width=0.8, color=discharge_color, alpha=0.7, label="Discharge (MWh)")
            ax_top.axhline(0, color="black", linewidth=1)
            ax_top.set_title(f"Dispatch (schedule {idx})")
            if col_idx == 0:
                ax_top.set_ylabel("Energy per hour (MWh)")
            # SOC trajectory
            ax_bot = axes[1, col_idx]
            ax_bot.step(hours, np.array(sched_df["soc_MWh"], dtype=float), where="mid", linewidth=3, color=soc_color)
            ax_bot.set_ylim(0, Emax_plot + 1)
            ax_bot.grid(True, alpha=0.3)
            ax_bot.set_title(f"SOC (schedule {idx})")
            ax_bot.set_xlabel("Hour")
            if col_idx == 0:
                ax_bot.set_ylabel("State of charge (MWh)")
        # Single legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        axes[0, 0].legend(handles, labels, loc="upper left")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def plot_aggregate_schedule(
        self,
        schedules: Sequence[ScenarioResult],
        *,
        figsize: Tuple[float, float] = (12, 8),
        Emax: Optional[float] = None,
        show: bool = True,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """Plot the average and standard deviation of dispatch and SOC across scenarios.

        This function aggregates the ``charge_MWh``, ``discharge_MWh`` and
        ``soc_MWh`` columns across all provided schedules, computes their means
        and standard deviations by hour and displays two subplots: one with
        bar charts for average charge (negative) and discharge (positive) with
        error bars, and another with the SOC mean along with a ±1 standard
        deviation band.

        Parameters
        ----------
        schedules : sequence of ScenarioResult
            Results returned by :meth:`solve`.
        figsize : tuple, optional
            Figure size.  Defaults to (12, 8).
        Emax : float, optional
            Battery capacity to set y‑limits on the SOC plot.  If omitted
            ``self.Emax`` is used.
        show : bool, optional
            If True (default), calls ``plt.show()``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The resulting figure.
        (ax_disp, ax_soc) : tuple of matplotlib.axes.Axes
            Axes for the dispatch and SOC plots.
        """
        if not schedules:
            raise ValueError("schedules must be a non‑empty sequence")
        # Concatenate all schedules with multiindex on scenario
        all_scheds = pd.concat(
            [res.schedule.assign(scenario=i) for i, res in enumerate(schedules)],
            ignore_index=True
        )
        # Compute mean and std by hour
        agg = (
            all_scheds
            .groupby("hour")[["charge_MWh", "discharge_MWh", "soc_MWh"]]
            .agg(["mean", "std"])
        )
        hours = agg.index
        # Prepare plots
        fig, (ax_disp, ax_soc) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        # Dispatch mean (charge negative) with std error bars
        charge_mean = -agg[("charge_MWh", "mean")]
        charge_std = agg[("charge_MWh", "std")]
        discharge_mean = agg[("discharge_MWh", "mean")]
        discharge_std = agg[("discharge_MWh", "std")]
        ax_disp.bar(hours, charge_mean, yerr=charge_std, width=0.8, color="#1f77b4", alpha=0.7, label="Charge (avg, neg)")
        ax_disp.bar(hours, discharge_mean, yerr=discharge_std, width=0.8, color="#ff7f0e", alpha=0.7, label="Discharge (avg)")
        ax_disp.axhline(0, color="black", linewidth=1)
        ax_disp.set_ylabel("Energy per hour (MWh)")
        ax_disp.set_title("Average dispatch with std error bars")
        ax_disp.legend(loc="upper left")
        # SOC mean ± std
        soc_mean = agg[("soc_MWh", "mean")]
        soc_std = agg[("soc_MWh", "std")]
        Emax_plot = float(self.Emax) if Emax is None else float(Emax)
        ax_soc.step(hours, soc_mean, where="mid", linewidth=3, color="#2ca02c", label="SOC mean")
        ax_soc.fill_between(hours, soc_mean - soc_std, soc_mean + soc_std, step="mid", color="#2ca02c", alpha=0.2, label="±1 std")
        ax_soc.set_ylabel("State of charge (MWh)")
        ax_soc.set_xlabel("Hour")
        ax_soc.set_ylim(0, max(Emax_plot, float((soc_mean + soc_std).max())) + 1)
        ax_soc.grid(True, alpha=0.3)
        ax_soc.set_title("Average SOC with std band")
        ax_soc.legend(loc="upper left")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, (ax_disp, ax_soc)