from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .pawparse import PawDataset

HA_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903


def features_from_ecut(
    df_ecut: pd.DataFrame,
    rpaw_bohr: float,
    L_bohr: float = 20.0,
) -> pd.DataFrame:
    """
    Enrich cutoff tiers with derived resolution and basis metrics.
    """
    if "Ecut_Ha" not in df_ecut.columns:
        raise KeyError("df_ecut must contain an 'Ecut_Ha' column")

    derived = df_ecut.copy()
    derived["Ecut_eV"] = derived["Ecut_Ha"] * HA_TO_EV
    derived["Gmax_bohr^-1"] = np.sqrt(2.0 * derived["Ecut_Ha"])
    derived["dx_bohr"] = np.pi / derived["Gmax_bohr^-1"]
    derived["dx_ang"] = derived["dx_bohr"] * BOHR_TO_ANG
    derived["plane_waves"] = (L_bohr**3) * (derived["Gmax_bohr^-1"] ** 3) / (6.0 * np.pi**2)
    derived["hardness_index"] = derived["Ecut_Ha"] / rpaw_bohr
    derived["supercell_edge_bohr"] = L_bohr
    derived["supercell_edge_ang"] = L_bohr * BOHR_TO_ANG
    return derived


def annotate_accuracy_labels(df: pd.DataFrame, delta: float | None, delta1: float | None) -> pd.DataFrame:
    """Attach Δ-metrics (if available) to a derived features frame."""
    annotated = df.copy()
    annotated["delta_metric"] = delta
    annotated["delta1_metric"] = delta1
    return annotated


def summarize_hardness(df_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hardness and cutoff tiers into an interview-friendly table."""
    required = {"tier", "Ecut_Ha", "hardness_index"}
    missing = required - set(df_features.columns)
    if missing:
        raise KeyError(f"df_features missing columns: {', '.join(sorted(missing))}")
    summary = df_features.loc[:, ["tier", "Ecut_Ha", "Ecut_eV", "hardness_index"]].copy()
    summary["Ecut_eV"] = summary["Ecut_Ha"] * HA_TO_EV
    return summary


def dataset_features(dataset: PawDataset, label: str | None = None, L_bohr: float = 20.0) -> pd.DataFrame:
    """
    Generate the derived feature table for a single dataset while carrying metadata.
    """
    features = features_from_ecut(dataset.df_ecut, dataset.meta["rpaw_bohr"], L_bohr=L_bohr)
    features["dataset_label"] = label or dataset.meta.get("xc_name", "unknown")
    features["xc_name"] = dataset.meta.get("xc_name")
    features["xc_type"] = dataset.meta.get("xc_type")
    features["rpaw_bohr"] = dataset.meta.get("rpaw_bohr")
    features["generator_name"] = dataset.meta.get("generator_name")
    features["delta_metric"] = dataset.meta.get("delta_metric")
    features["delta1_metric"] = dataset.meta.get("delta1_metric")
    return features


def compare_datasets(
    datasets: Mapping[str, PawDataset] | Iterable[tuple[str, PawDataset]],
    L_bohr: float = 20.0,
) -> pd.DataFrame:
    """
    Assemble a comparison table across multiple PAW datasets.
    """
    rows = []
    items = datasets.items() if isinstance(datasets, Mapping) else datasets
    for label, dataset in items:
        df = dataset_features(dataset, label=label, L_bohr=L_bohr)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
