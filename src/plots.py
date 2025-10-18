from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_ae_energy(df_energy: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Stacked bar of all-electron energy components."""
    required_cols = ["kinetic_Ha", "xc_Ha", "electrostatic_Ha", "total_Ha"]
    if not set(required_cols).issubset(df_energy.columns):
        missing = set(required_cols) - set(df_energy.columns)
        raise KeyError(f"df_energy missing columns: {', '.join(sorted(missing))}")

    ax = ax or plt.gca()
    components = ["kinetic_Ha", "xc_Ha", "electrostatic_Ha"]
    values = df_energy.iloc[0][components]
    ax.bar(components, values, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.axhline(df_energy.iloc[0]["total_Ha"], color="#8172b3", linestyle="--", label="total")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title("All-Electron Energy Budget")
    ax.legend()
    return ax


def plot_radial_grid(df_grid: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot radial grid points versus index."""
    if not {"index", "r_bohr"}.issubset(df_grid.columns):
        missing = {"index", "r_bohr"} - set(df_grid.columns)
        raise KeyError(f"df_grid missing columns: {', '.join(sorted(missing))}")
    ax = ax or plt.gca()
    ax.plot(df_grid["index"], df_grid["r_bohr"], lw=1.0)
    ax.set_xlabel("Grid index")
    ax.set_ylabel("Radius (Bohr)")
    ax.set_title("Radial Grid (log spacing)")
    return ax


def plot_radial_spacing(df_grid: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot radial spacing Δr versus r."""
    required = {"r_bohr", "delta_r_bohr"}
    if not required.issubset(df_grid.columns):
        missing = required - set(df_grid.columns)
        raise KeyError(f"df_grid missing columns: {', '.join(sorted(missing))}")
    ax = ax or plt.gca()
    ax.plot(df_grid["r_bohr"], df_grid["delta_r_bohr"], lw=1.0)
    ax.set_xlabel("Radius (Bohr)")
    ax.set_ylabel("Δr (Bohr)")
    ax.set_yscale("log")
    ax.set_title("Radial Grid Resolution Profile")
    return ax


def plot_cost_vs_cutoff(df_features: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Visualize how resolution and plane-wave count evolve with cutoff."""
    required = {"tier", "Ecut_Ha", "dx_ang", "plane_waves"}
    if not required.issubset(df_features.columns):
        missing = required - set(df_features.columns)
        raise KeyError(f"df_features missing columns: {', '.join(sorted(missing))}")
    ax = ax or plt.gca()
    tiers = df_features.sort_values("Ecut_Ha")
    ax.plot(tiers["Ecut_Ha"], tiers["plane_waves"], marker="o", label="Plane waves")
    ax.set_xlabel("Cutoff (Ha)")
    ax.set_ylabel("Estimated plane waves")
    ax2 = ax.twinx()
    ax2.plot(tiers["Ecut_Ha"], tiers["dx_ang"], marker="s", color="#55a868", label="Δx (Å)")
    ax2.set_ylabel("Real-space grid spacing (Å)")
    ax.set_title("Cost vs. Resolution")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    return ax


def plot_cost_vs_cutoff_compare(df_features: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Compare cost vs cutoff behaviour across multiple datasets."""
    required = {"dataset_label", "Ecut_Ha", "plane_waves", "dx_ang"}
    if not required.issubset(df_features.columns):
        missing = required - set(df_features.columns)
        raise KeyError(f"df_features missing columns: {', '.join(sorted(missing))}")
    ax = ax or plt.gca()
    cmap = plt.get_cmap("tab10")
    ax2 = ax.twinx()
    for idx, (label, group) in enumerate(df_features.groupby("dataset_label")):
        color = cmap(idx % 10)
        ordered = group.sort_values("Ecut_Ha")
        ax.plot(
            ordered["Ecut_Ha"],
            ordered["plane_waves"],
            marker="o",
            color=color,
            label=f"{label} plane waves",
        )
        ax2.plot(
            ordered["Ecut_Ha"],
            ordered["dx_ang"],
            marker="s",
            linestyle="--",
            color=color,
            label=f"{label} Δx",
        )
    ax.set_xlabel("Cutoff (Ha)")
    ax.set_ylabel("Estimated plane waves")
    ax2.set_ylabel("Real-space grid spacing (Å)")
    ax.set_title("Cost vs. Resolution (Across PAWs)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    return ax
