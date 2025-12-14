"""Utilities for loading time‑series wind/price data and generating common plots.

This module defines a :class:`PlotDataUtils` class that encapsulates the logic
for reading the input CSV used in the tutorials, computing summary statistics
across wind production scenarios, and generating the various line plots and
fan charts shown in the example notebook.  Each public method either
returns a pandas object with computed values or renders a plot on the
provided Matplotlib axes.

The class is designed so that you can call only the pieces you need.  For
example, you can load the data and compute summary statistics without
plotting, or you can create all of the figures from a single instantiated
object.

Example
-------

::

    from pathlib import Path
    from main.plot_data_utils import PlotDataUtils

    # Construct a utility for the provided CSV
    utils = PlotDataUtils(Path('data/input_data.csv'))
    df = utils.load_data()
    df = utils.compute_wind_stats(df)
    summary = utils.wind_energy_summary(df)
    corr = utils.price_wind_correlation(df)
    utils.plot_price_curve(df)             # price vs hour line plot
    utils.plot_wind_scenarios(df)          # all wind scenarios and mean
    utils.plot_wind_fan_chart(df)          # quantile bands of wind production
    rev_df, rev_mean, rev_quant_df = utils.compute_revenue_stats(df)
    utils.plot_revenue_overlay(df, rev_df, rev_mean)
    utils.plot_revenue_fan_chart(df, rev_quant_df)

"""

from __future__ import annotations

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable, Tuple, List


class PlotDataUtils:
    """High‑level helper class to load and plot wind production and price data.

    Parameters
    ----------
    data_path : Path or str
        Location of the CSV file to read.  If the path does not exist, the
        constructor will look for a file with the same name in the parent
        directory of the current working directory.
    sns_context : str, optional
        Seaborn context to set for plotting.  Defaults to ``"talk"``.
    sns_style : str, optional
        Seaborn style to use for plotting.  Defaults to ``"whitegrid"``.

    Attributes
    ----------
    scenario_cols : List[str]
        Column names in the loaded DataFrame that correspond to individual
        wind forecast scenarios.
    """

    def __init__(self, data_path: Path | str, *, sns_context: str = "talk", sns_style: str = "whitegrid") -> None:
        # Determine and store the path to the CSV file
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            # Try resolving relative to parent directory (e.g. when running from test/)
            candidate = Path.cwd().parent / self.data_path.name
            if candidate.exists():
                self.data_path = candidate
        # Set default seaborn configuration
        sns.set_theme(style=sns_style, context=sns_context)
        # Will be populated after loading
        self.scenario_cols: List[str] = []

    # ------------------------------------------------------------------
    # Data loading and preprocessing
    # ------------------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """Load the CSV file into a DataFrame.

        Returns
        -------
        pandas.DataFrame
            The raw data with no computed statistics.
        """
        df = pd.read_csv(self.data_path)
        # Identify columns that begin with "scenario_"
        self.scenario_cols = [c for c in df.columns if c.startswith("scenario_")]
        return df

    def compute_wind_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mean and quantile columns for wind scenarios to the DataFrame.

        The method computes the mean, 10th percentile and 90th percentile across
        all scenario columns for each row and stores the results in new
        columns: ``mean_wind``, ``p10_wind`` and ``p90_wind``.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame returned by :meth:`load_data`.

        Returns
        -------
        pandas.DataFrame
            The original DataFrame with three new columns.
        """
        if not self.scenario_cols:
            raise ValueError("No scenario columns identified; call load_data() first.")
        df = df.copy()
        df["mean_wind"] = df[self.scenario_cols].mean(axis=1)
        df["p10_wind"] = df[self.scenario_cols].quantile(0.10, axis=1)
        df["p90_wind"] = df[self.scenario_cols].quantile(0.90, axis=1)
        return df

    def wind_energy_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate wind production and price statistics across all hours.

        The resulting DataFrame contains a single row with columns:

        - ``expected_MWh`` – sum of ``mean_wind``
        - ``p10_MWh`` – sum of ``p10_wind``
        - ``p90_MWh`` – sum of ``p90_wind``
        - ``price_mean`` – mean of the ``price`` column
        - ``price_std`` – standard deviation of the ``price`` column
        - ``price_min`` – minimum ``price``
        - ``price_max`` – maximum ``price``

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with ``mean_wind``, ``p10_wind`` and ``p90_wind`` columns.

        Returns
        -------
        pandas.DataFrame
            A single‑row DataFrame summarizing wind energy and price statistics.
        """
        required_cols = {"mean_wind", "p10_wind", "p90_wind", "price"}
        if not required_cols.issubset(df.columns):
            missing = ", ".join(sorted(required_cols - set(df.columns)))
            raise ValueError(f"Missing required columns for summary: {missing}")
        summary = pd.DataFrame({
            "expected_MWh": [df["mean_wind"].sum()],
            "p10_MWh": [df["p10_wind"].sum()],
            "p90_MWh": [df["p90_wind"].sum()],
            "price_mean": [df["price"].mean()],
            "price_std": [df["price"].std()],
            "price_min": [df["price"].min()],
            "price_max": [df["price"].max()],
        })
        return summary

    def price_wind_correlation(self, df: pd.DataFrame) -> float:
        """Compute the correlation between mean wind and price.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing ``mean_wind`` and ``price`` columns.

        Returns
        -------
        float
            The Pearson correlation coefficient between ``mean_wind`` and ``price``.
        """
        if not {"mean_wind", "price"}.issubset(df.columns):
            raise ValueError("DataFrame must contain 'mean_wind' and 'price' columns")
        return df[["mean_wind", "price"]].corr().iloc[0, 1]

    # ------------------------------------------------------------------
    # Plotting methods
    # ------------------------------------------------------------------
    def plot_price_curve(self, df: pd.DataFrame, ax: plt.Axes | None = None, *, color: str = "#1f77b4") -> plt.Axes:
        """Draw the day‑ahead price curve as a line plot.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing ``hour`` and ``price`` columns.
        ax : matplotlib.axes.Axes, optional
            Existing axis to draw on.  If ``None`` (default), a new figure
            and axis are created.
        color : str, optional
            Colour for the price curve.  Defaults to the first colour in the
            Matplotlib cycle.

        Returns
        -------
        matplotlib.axes.Axes
            The axis on which the line plot was drawn.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=df, x="hour", y="price", marker="o", color=color, linewidth=3, ax=ax)
        ax.set_ylabel("Market price (EUR/MWh)")
        ax.set_xlabel("Hour of day")
        ax.set_title("Day‑ahead price curve")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return ax

    def plot_wind_scenarios(self, df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot all wind production scenarios together with their mean.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing ``hour``, scenario columns and ``mean_wind``.
        ax : matplotlib.axes.Axes, optional
            Existing axis to draw on.  If ``None`` (default) a new figure
            is created.

        Returns
        -------
        matplotlib.axes.Axes
            The axis on which the plot was drawn.
        """
        if not self.scenario_cols:
            raise ValueError("No scenario columns identified; call load_data() first.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))
        for col in self.scenario_cols:
            ax.plot(df["hour"], df[col], alpha=0.35, linewidth=2)
        ax.plot(df["hour"], df["mean_wind"], color="black", linewidth=3, label="Scenario mean")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Wind production (MWh)")
        ax.set_title(f"Wind production forecasts ({len(self.scenario_cols)} equiprobable scenarios)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        return ax

    def compute_wind_quantiles(self, df: pd.DataFrame, quantiles: Iterable[float] = (0.10, 0.25, 0.5, 0.75, 0.90)) -> pd.DataFrame:
        """Compute wind production quantiles for each hour.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing scenario columns.
        quantiles : Iterable[float], optional
            Quantile levels to compute for each hour.  Defaults to
            ``(0.10, 0.25, 0.5, 0.75, 0.90)``.

        Returns
        -------
        pandas.DataFrame
            DataFrame whose columns are the requested quantiles and whose
            index matches the ``hour`` column of ``df``.
        """
        if not self.scenario_cols:
            raise ValueError("No scenario columns identified; call load_data() first.")
        # Compute each quantile across the scenario columns rowwise
        quant_df = pd.DataFrame({q: df[self.scenario_cols].quantile(q, axis=1) for q in quantiles})
        return quant_df

    def plot_wind_fan_chart(self, df: pd.DataFrame, quant_df: pd.DataFrame | None = None, ax: plt.Axes | None = None) -> plt.Axes:
        """Draw a fan chart showing wind production quantile bands.

        If ``quant_df`` is not provided it will be computed on the fly using
        :meth:`compute_wind_quantiles` with its default settings.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing ``hour`` and scenario columns.
        quant_df : pandas.DataFrame, optional
            Precomputed quantile DataFrame with columns 0.10, 0.25, 0.5, 0.75, 0.90.  If
            provided, must align with ``df`` by index.
        ax : matplotlib.axes.Axes, optional
            Existing axis to draw on.  A new figure is created if ``None``.

        Returns
        -------
        matplotlib.axes.Axes
            The axis on which the fan chart was drawn.
        """
        if quant_df is None:
            quant_df = self.compute_wind_quantiles(df)
        # Ensure expected quantiles are present
        required = {0.1, 0.25, 0.5, 0.75, 0.9}
        if not required.issubset(set(quant_df.columns)):
            missing = ", ".join(str(q) for q in sorted(required - set(quant_df.columns)))
            raise ValueError(f"Quantile DataFrame missing required columns: {missing}")
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))
        # Plot median and fill quantile bands
        ax.plot(df["hour"], quant_df[0.5], color="black", linewidth=3, label="Median")
        ax.fill_between(df["hour"], quant_df[0.25], quant_df[0.75], color="#1f77b4", alpha=0.25, label="25–75% band")
        ax.fill_between(df["hour"], quant_df[0.1], quant_df[0.9], color="#1f77b4", alpha=0.12, label="10–90% band")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Wind production (MWh)")
        ax.set_title("Wind production fan chart")
        ax.legend(loc="upper right")
        plt.tight_layout()
        return ax

    # ------------------------------------------------------------------
    # Revenue calculations and plots
    # ------------------------------------------------------------------
    def compute_revenue_stats(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Compute revenue per scenario and corresponding statistics.

        The revenue for each scenario is obtained by multiplying each wind
        scenario by the corresponding hourly price.  The return values are:

        - ``rev_df`` – DataFrame of the same shape as the wind scenario
          DataFrame containing revenues (EUR) per scenario per hour.
        - ``rev_mean`` – pandas Series giving the mean revenue across scenarios
          for each hour.
        - ``rev_quant_df`` – DataFrame of quantiles (0.10, 0.25, 0.5, 0.75, 0.90)
          computed across the scenarios for each hour.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing scenario columns and ``price``.

        Returns
        -------
        tuple of (DataFrame, Series, DataFrame)
            The per–scenario revenue matrix, its mean across scenarios and
            quantiles of revenue across scenarios.
        """
        if not self.scenario_cols:
            raise ValueError("No scenario columns identified; call load_data() first.")
        # Multiply each scenario column by the price column row‑wise
        rev_df = df[self.scenario_cols].mul(df["price"], axis=0)
        rev_mean = rev_df.mean(axis=1)
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        rev_quant_df = pd.DataFrame({q: rev_df.quantile(q, axis=1) for q in quantiles})
        return rev_df, rev_mean, rev_quant_df

    def price_revenue_correlations(self, df: pd.DataFrame, rev_df: pd.DataFrame, rev_mean: pd.Series) -> Tuple[float, pd.Series]:
        """Compute correlations between price and revenue.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the ``price`` column.
        rev_df : pandas.DataFrame
            Revenue per scenario per hour as returned by :meth:`compute_revenue_stats`.
        rev_mean : pandas.Series
            Mean revenue per hour as returned by :meth:`compute_revenue_stats`.

        Returns
        -------
        tuple
            A tuple ``(corr_mean, corr_by_scenario)`` where ``corr_mean`` is
            the correlation between price and mean revenue, and
            ``corr_by_scenario`` is a Series of correlations computed for each
            scenario separately.
        """
        corr_mean = rev_mean.corr(df["price"])
        corr_by_scenario = rev_df.apply(lambda s: s.corr(df["price"]))
        return corr_mean, corr_by_scenario

    def plot_revenue_overlay(self, df: pd.DataFrame, rev_df: pd.DataFrame, rev_mean: pd.Series, ax1: plt.Axes | None = None, ax2: plt.Axes | None = None) -> Tuple[plt.Axes, plt.Axes]:
        """Overlay revenue per scenario with the price curve on a dual‑axis plot.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing ``hour`` and ``price``.
        rev_df : pandas.DataFrame
            Revenue per scenario per hour.
        rev_mean : pandas.Series
            Mean revenue per hour.
        ax1, ax2 : matplotlib.axes.Axes, optional
            If provided, the function will draw on these axes instead of
            creating new ones.  ``ax1`` is used for revenue lines and ``ax2``
            for the price line.  When ``ax2`` is ``None`` but ``ax1`` is not,
            ``ax2`` is created by calling :meth:`matplotlib.axes.Axes.twinx`.

        Returns
        -------
        tuple of (Axes, Axes)
            The axes on which the revenue and price have been plotted.
        """
        if ax1 is None:
            fig, ax1 = plt.subplots(figsize=(12, 5))
        if ax2 is None:
            ax2 = ax1.twinx()
        # Plot each scenario's revenue
        for col in rev_df.columns:
            ax1.plot(df["hour"], rev_df[col], alpha=0.25, linewidth=1.5)
        # Plot the mean revenue
        ax1.plot(df["hour"], rev_mean, color="black", linewidth=3, label="Scenario mean revenue")
        ax1.set_xlabel("Hour of day")
        ax1.set_ylabel("Revenue (EUR)")
        # Plot price on second axis
        ax2.plot(df["hour"], df["price"], color="crimson", linewidth=2.5, marker="o", label="Price")
        ax2.set_ylabel("Market price (EUR/MWh)")
        ax1.set_title("Hourly wind revenue vs price")
        # Combine legends from both axes
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        return ax1, ax2

    def plot_revenue_fan_chart(self, df: pd.DataFrame, rev_quant_df: pd.DataFrame, ax1: plt.Axes | None = None, ax2: plt.Axes | None = None) -> Tuple[plt.Axes, plt.Axes]:
        """Plot revenue quantile bands alongside the price curve.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing ``hour`` and ``price``.
        rev_quant_df : pandas.DataFrame
            Revenue quantiles across scenarios.  Must contain columns
            ``0.10``, ``0.25``, ``0.50``, ``0.75`` and ``0.90``.
        ax1, ax2 : matplotlib.axes.Axes, optional
            Axes for revenue quantiles and price.  If provided, the function
            draws on them; otherwise new axes are created.

        Returns
        -------
        tuple of (Axes, Axes)
            The axes used for plotting revenue quantiles and price.
        """
        # Validate quantile columns
        required = {0.10, 0.25, 0.50, 0.75, 0.90}
        if not required.issubset(set(rev_quant_df.columns)):
            missing = ", ".join(str(q) for q in sorted(required - set(rev_quant_df.columns)))
            raise ValueError(f"rev_quant_df missing required quantile columns: {missing}")
        if ax1 is None:
            fig, ax1 = plt.subplots(figsize=(12, 5))
        if ax2 is None:
            ax2 = ax1.twinx()
        # Plot median and bands for revenue
        ax1.plot(df["hour"], rev_quant_df[0.50], color="black", linewidth=3, label="Median revenue")
        ax1.fill_between(df["hour"], rev_quant_df[0.25], rev_quant_df[0.75], color="#1f77b4", alpha=0.25, label="25–75% band")
        ax1.fill_between(df["hour"], rev_quant_df[0.10], rev_quant_df[0.90], color="#1f77b4", alpha=0.12, label="10–90% band")
        ax1.set_xlabel("Hour of day")
        ax1.set_ylabel("Revenue (EUR)")
        # Plot price on secondary axis
        ax2.plot(df["hour"], df["price"], color="crimson", linewidth=2.5, marker="o", label="Price")
        ax2.set_ylabel("Market price (EUR/MWh)")
        ax1.set_title("Hourly revenue quantiles vs price")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        return ax1, ax2
    
    def wind_revenue(self, df: pd.DataFrame) -> Tuple[pd.Series, float, float, float]:
        """Compute total wind revenue per scenario and print summary statistics.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing scenario columns and ``price``.

        Returns
        -------
        tuple
            Tuple of ``(wind_rev_by_s, wind_rev_exp, wind_rev_min, wind_rev_max)`` where
            the first element is total revenue per scenario and the rest are
            summary statistics across scenarios.
        """
        scenario_cols = self.scenario_cols or [c for c in df.columns if c.startswith("scenario_")]
        if not scenario_cols:
            raise ValueError("No scenario columns identified; call load_data() first.")
        wind_rev_by_s = (df[scenario_cols].mul(df["price"], axis=0)).sum(axis=0)
        wind_rev_exp = wind_rev_by_s.mean()
        wind_rev_var = wind_rev_by_s.std()
        wind_rev_min = wind_rev_by_s.min()
        wind_rev_max = wind_rev_by_s.max()

        return wind_rev_by_s, wind_rev_exp, wind_rev_min, wind_rev_max, wind_rev_var
