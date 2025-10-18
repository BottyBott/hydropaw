from pathlib import Path

from src.pawparse import parse_paw_xml


DATA_PATH = Path("data/H.GGA_PBE-JTH.xml")


def test_parse_meta_fields():
    dataset = parse_paw_xml(DATA_PATH)
    assert dataset.meta["symbol"] == "H"
    assert abs(dataset.meta["Z"] - 1.0) < 1e-8
    assert dataset.meta["xc_name"] == "PBE"
    assert dataset.meta["generator_name"].startswith("atompaw")
    assert dataset.meta["rpaw_bohr"] > 0.9


def test_energy_and_cutoff_tables():
    dataset = parse_paw_xml(DATA_PATH)
    assert set(dataset.df_energy.columns) == {
        "kinetic_Ha",
        "xc_Ha",
        "electrostatic_Ha",
        "total_Ha",
    }
    assert dataset.df_ecut.set_index("tier")["Ecut_Ha"].to_dict() == {
        "low": 17.5,
        "medium": 20.0,
        "high": 25.0,
    }


def test_valence_and_grid_shapes():
    dataset = parse_paw_xml(DATA_PATH)
    assert len(dataset.df_valence) >= 1
    assert {"r_bohr", "delta_r_bohr"}.issubset(dataset.df_grid.columns)
    assert len(dataset.df_grid) == dataset.grid_params["iend"] - dataset.grid_params["istart"] + 1


def test_second_file_metadata():
    second = parse_paw_xml(Path("data/H.LDA_PW-JTH.xml"))
    assert second.meta["xc_name"] == "PW"
    assert second.meta["xc_type"] == "LDA"
