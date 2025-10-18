from pathlib import Path

import pandas as pd

from src.features import compare_datasets, dataset_features
from src.pawparse import parse_multiple


PATHS = [
    Path("data/H.GGA_PBE-JTH.xml"),
    Path("data/H.LDA_PW-JTH.xml"),
]


def test_dataset_features_labels():
    datasets = parse_multiple(PATHS)
    gga = dataset_features(datasets[PATHS[0].resolve()], label="PBE/JTH")
    assert (gga["dataset_label"] == "PBE/JTH").all()
    assert gga["xc_name"].iat[0] == "PBE"
    assert "plane_waves" in gga


def test_compare_datasets_stack_shape():
    datasets = parse_multiple(PATHS)
    comparison = compare_datasets({path.stem: datasets[path.resolve()] for path in PATHS})
    assert isinstance(comparison, pd.DataFrame)
    assert set(comparison["dataset_label"]) == {"H.GGA_PBE-JTH", "H.LDA_PW-JTH"}
    assert comparison.shape[0] == 6  # 3 tiers × 2 datasets
