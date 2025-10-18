from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from lxml import etree


class PawParseError(RuntimeError):
    """Raised when a required PAW XML field is missing or malformed."""


def _coerce_float(value: str | float | None) -> float:
    if value is None:
        raise PawParseError("Expected float attribute, received None")
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except ValueError as exc:
        raise PawParseError(f"Cannot convert '{value}' to float") from exc


def _coerce_int(value: str | int | None) -> int:
    if value is None:
        raise PawParseError("Expected int attribute, received None")
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise PawParseError(f"Cannot convert '{value}' to int") from exc


def _get_required(node: etree._Element, tag: str) -> etree._Element:
    elem = node.find(tag)
    if elem is None:
        raise PawParseError(f"Missing required element <{tag}> in {node.tag}")
    return elem


def _parse_valence_states(parent: etree._Element) -> pd.DataFrame:
    states: List[Dict[str, object]] = []
    for state in parent.findall("state"):
        row: Dict[str, object] = {}
        for key in ("id",):
            if state.get(key) is not None:
                row[key] = state.get(key).strip()
        for key in ("n", "l"):
            if state.get(key) is not None:
                row[key] = _coerce_int(state.get(key))
        for key in ("f", "rc", "e"):
            if state.get(key) is not None:
                row[key] = _coerce_float(state.get(key))
        states.append(row)
    return pd.DataFrame(states)


def _parse_radial_grid(grid: etree._Element) -> Tuple[Dict[str, float | str], pd.DataFrame]:
    params: Dict[str, float | str] = {
        "equation": grid.get("eq", "").strip(),
        "id": grid.get("id", "").strip(),
        "a": _coerce_float(grid.get("a")),
        "d": _coerce_float(grid.get("d")),
        "istart": _coerce_int(grid.get("istart")),
        "iend": _coerce_int(grid.get("iend")),
    }
    values_node = _get_required(grid, "values")
    text_values = values_node.text or ""
    r_values = [_coerce_float(val) for val in text_values.split()]
    df_grid = pd.DataFrame(
        {
            "index": range(params["istart"], params["istart"] + len(r_values)),
            "r_bohr": r_values,
        }
    )
    df_grid["delta_r_bohr"] = df_grid["r_bohr"].diff().fillna(0.0)
    return params, df_grid


@dataclass
class PawDataset:
    meta: Dict[str, object]
    df_energy: pd.DataFrame
    df_ecut: pd.DataFrame
    df_valence: pd.DataFrame
    df_grid: pd.DataFrame
    grid_params: Dict[str, float | str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "meta": self.meta,
            "df_energy": self.df_energy,
            "df_ecut": self.df_ecut,
            "df_valence": self.df_valence,
            "df_grid": self.df_grid,
            "grid_params": self.grid_params,
        }


def parse_paw_xml(path: str | Path) -> PawDataset:
    """Parse a PAW XML file into tidy pandas tables and metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PAW XML not found: {path}")

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    first_tag = raw_text.find("<")
    if first_tag == -1:
        raise PawParseError(f"No XML start tag found in {path}")
    try:
        root = etree.fromstring(raw_text[first_tag:].encode("utf-8"))
    except etree.XMLSyntaxError as exc:
        raise PawParseError(f"Failed to parse XML from {path}: {exc}") from exc

    atom = _get_required(root, "atom")
    paw_radius = _get_required(root, "paw_radius")
    ecut = _get_required(root, "pw_ecut")
    xc = _get_required(root, "xc_functional")
    generator = _get_required(root, "generator")
    ae_energy = _get_required(root, "ae_energy")
    valence_states = _get_required(root, "valence_states")
    radial_grid = _get_required(root, "radial_grid")

    meta: Dict[str, object] = {
        "symbol": atom.get("symbol", "").strip(),
        "Z": _coerce_float(atom.get("Z")),
        "core_electrons": _coerce_float(atom.get("core")),
        "valence_electrons": _coerce_float(atom.get("valence")),
        "xc_type": xc.get("type", "").strip(),
        "xc_name": xc.get("name", "").strip(),
        "generator_name": generator.get("name", "").strip(),
        "generator_kind": generator.get("type", "").strip(),
        "rpaw_bohr": _coerce_float(paw_radius.get("rc")),
    }

    core_energy = root.find("core_energy")
    if core_energy is not None and core_energy.get("kinetic") is not None:
        meta["core_kinetic_energy_ha"] = _coerce_float(core_energy.get("kinetic"))

    for tag in ("delta_metric", "delta1_metric"):
        node = root.find(tag)
        if node is not None and node.text:
            try:
                meta[tag] = _coerce_float(node.text)
            except PawParseError:
                meta[tag] = node.text.strip()

    df_energy = pd.DataFrame(
        [
            {
                "kinetic_Ha": _coerce_float(ae_energy.get("kinetic")),
                "xc_Ha": _coerce_float(ae_energy.get("xc")),
                "electrostatic_Ha": _coerce_float(ae_energy.get("electrostatic")),
                "total_Ha": _coerce_float(ae_energy.get("total")),
            }
        ]
    )

    df_ecut = pd.DataFrame(
        [
            {"tier": "low", "Ecut_Ha": _coerce_float(ecut.get("low"))},
            {"tier": "medium", "Ecut_Ha": _coerce_float(ecut.get("medium"))},
            {"tier": "high", "Ecut_Ha": _coerce_float(ecut.get("high"))},
        ]
    )

    df_valence = _parse_valence_states(valence_states)
    grid_params, df_grid = _parse_radial_grid(radial_grid)

    return PawDataset(
        meta=meta,
        df_energy=df_energy,
        df_ecut=df_ecut,
        df_valence=df_valence,
        df_grid=df_grid,
        grid_params=grid_params,
    )


def parse_multiple(paths: Iterable[str | Path]) -> Dict[Path, PawDataset]:
    """Parse multiple PAW datasets, keyed by their resolved paths."""
    datasets: Dict[Path, PawDataset] = {}
    for path in paths:
        dataset = parse_paw_xml(path)
        datasets[Path(path).resolve()] = dataset
    return datasets
