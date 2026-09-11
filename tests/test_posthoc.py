from pathlib import Path

import torch

from rule_grounded_sgg.data import load_config, load_graphs
from rule_grounded_sgg.posthoc_validation import apply_posthoc_validation


ROOT = Path(__file__).resolve().parents[1]


def test_posthoc_removes_illegal_foreground_prediction():
    config = load_config(ROOT / "configs/paper_default.yaml")
    graph = load_graphs(ROOT / "examples/synthetic_graphs.json", config)[0]
    predicates = config["classes"]["predicates"]
    predictions = torch.full((6,), predicates.index("none"), dtype=torch.long)
    predictions[0] = predicates.index("performing")  # Worker -> Helmet is illegal.
    validated, keep = apply_posthoc_validation(predictions, graph, config)
    assert not bool(keep[0])
    assert int(validated[0]) == predicates.index("none")
