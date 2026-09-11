from pathlib import Path

import torch
from torch import nn

from rule_grounded_sgg.data import geometry_and_evidence, load_config, load_graphs
from rule_grounded_sgg.metrics import evaluate
from rule_grounded_sgg.posthoc_validation import (
    aggregate_worker_relations,
    relation_validity,
)


ROOT = Path(__file__).resolve().parents[1]


def test_contact_evidence_combines_overlap_proximity_and_layout():
    # A small tool fully contained in a worker box must retain strong local-overlap evidence.
    _, contained = geometry_and_evidence(
        [100, 100, 300, 450],
        [180, 220, 220, 260],
        (640, 480),
    )
    assert contained[7] > 0.02

    # Close but non-overlapping boxes retain a weak continuous proximity-layout cue
    # below the performing gate; a far pair has no contact evidence.
    _, close = geometry_and_evidence(
        [100, 100, 200, 300],
        [205, 120, 305, 320],
        (640, 480),
    )
    _, far = geometry_and_evidence(
        [10, 10, 80, 100],
        [520, 350, 620, 460],
        (640, 480),
    )
    assert 0.0 < close[7] < 0.02
    assert far[7] == 0.0


class _FixedRelationModel(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, graph):
        return {"relation_logits": self.fixed_logits.to(graph["pair_index"].device)}


def test_posthoc_filtering_changes_topk_candidate_set():
    config = load_config(ROOT / "configs/paper_default.yaml")
    graph = load_graphs(ROOT / "examples/synthetic_graphs.json", config)[0]
    predicates = {name: i for i, name in enumerate(config["classes"]["predicates"])}

    logits = torch.full((6, 5), -5.0)
    logits[:, predicates["none"]] = 5.0
    # Edge 0 is Worker -> Helmet. Make an illegal performing prediction the
    # highest foreground candidate in the image.
    logits[0, predicates["performing"]] = 4.0
    logits[0, predicates["none"]] = 0.0
    # Edge 1 is the correctly labelled Worker -> Hazardous_area into relation.
    logits[1, predicates["into"]] = 3.0
    logits[1, predicates["none"]] = 0.0

    model = _FixedRelationModel(logits)
    raw = evaluate(model, [graph], config, torch.device("cpu"), posthoc=False)
    validated = evaluate(model, [graph], config, torch.device("cpu"), posthoc=True)

    assert raw["R@1"] == 0.0
    assert validated["R@1"] == 50.0
    assert validated["mR@1"] > raw["mR@1"]


def test_interaction_near_requires_proximity_or_layout_support():
    config = load_config(ROOT / "configs/paper_default.yaml")
    graph = load_graphs(ROOT / "examples/synthetic_graphs.json", config)[0]
    entities = {name: i for i, name in enumerate(config["classes"]["entities"])}
    predicates = {name: i for i, name in enumerate(config["classes"]["predicates"])}

    graph["object_categories"] = graph["object_categories"].clone()
    graph["object_categories"][1] = entities["Tool"]
    graph["spatial_evidence"] = graph["spatial_evidence"].clone()
    graph["pair_geometry"] = graph["pair_geometry"].clone()

    predictions = torch.full((6,), predicates["none"], dtype=torch.long)
    predictions[0] = predicates["near"]  # Worker -> Tool

    graph["spatial_evidence"][0, 0] = 0.8
    graph["spatial_evidence"][0, 1] = 0.8
    graph["spatial_evidence"][0, 5] = 0.8
    graph["spatial_evidence"][0, 7] = 0.0
    assert not bool(relation_validity(predictions, graph, config)[0])

    graph["spatial_evidence"][0, 0] = 0.05
    graph["spatial_evidence"][0, 1] = 0.05
    graph["spatial_evidence"][0, 5] = 0.01
    assert bool(relation_validity(predictions, graph, config)[0])


def test_worker_centered_aggregation_returns_relation_groups_and_states():
    config = load_config(ROOT / "configs/paper_default.yaml")
    graph = load_graphs(ROOT / "examples/synthetic_graphs.json", config)[0]
    predicates = {name: i for i, name in enumerate(config["classes"]["predicates"])}

    predictions = torch.full((6,), predicates["none"], dtype=torch.long)
    predictions[0] = predicates["wear"]
    predictions[1] = predicates["into"]
    worker = aggregate_worker_relations(predictions, graph, config)[0]

    assert worker["ppe"] == [0]
    assert worker["zone"] == [1]
    assert worker["interaction"] == []
    assert worker["states"] == {
        "ppe": "wear",
        "zone": "into",
        "interaction": "none",
    }
