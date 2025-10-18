from pathlib import Path

import pytest

from src.features import features_from_ecut, HA_TO_EV, BOHR_TO_ANG
from src.pawparse import parse_paw_xml


DATA_PATH = Path("data/H.GGA_PBE-JTH.xml")


@pytest.fixture(scope="module")
def dataset():
    return parse_paw_xml(DATA_PATH)


def test_features_include_expected_columns(dataset):
    feats = features_from_ecut(dataset.df_ecut, dataset.meta["rpaw_bohr"], L_bohr=20.0)
    expected = {
        "tier",
        "Ecut_Ha",
        "Ecut_eV",
        "Gmax_bohr^-1",
        "dx_bohr",
        "dx_ang",
        "plane_waves",
        "hardness_index",
        "supercell_edge_bohr",
        "supercell_edge_ang",
    }
    assert expected.issubset(feats.columns)


def test_feature_values_medium_cutoff(dataset):
    feats = features_from_ecut(dataset.df_ecut, dataset.meta["rpaw_bohr"], L_bohr=20.0)
    medium = feats.set_index("tier").loc["medium"]
    assert medium["Ecut_eV"] == pytest.approx(20.0 * HA_TO_EV, rel=1e-10)
    assert medium["Gmax_bohr^-1"] == pytest.approx((40.0) ** 0.5, rel=1e-10)
    assert medium["dx_bohr"] == pytest.approx(3.141592653589793 / medium["Gmax_bohr^-1"], rel=1e-10)
    assert medium["dx_ang"] == pytest.approx(medium["dx_bohr"] * BOHR_TO_ANG, rel=1e-10)
    assert medium["supercell_edge_ang"] == pytest.approx(20.0 * BOHR_TO_ANG, rel=1e-12)
    assert medium["hardness_index"] == pytest.approx(20.0 / dataset.meta["rpaw_bohr"], rel=1e-8)
