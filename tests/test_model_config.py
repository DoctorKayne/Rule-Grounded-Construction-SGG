from pathlib import Path

import torch

from rule_grounded_sgg.data import load_config, load_graphs
from rule_grounded_sgg.losses import compute_joint_loss
from rule_grounded_sgg.model import RuleGroundedSGG, TransformerReasoner


ROOT = Path(__file__).resolve().parents[1]


def trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def test_default_model_configuration():
    config = load_config(ROOT / "configs/paper_default.yaml")
    assert config["classes"]["entities"] == [
        "Worker",
        "Helmet",
        "Safety_vest",
        "Safety_belt",
        "Hazardous_area",
        "Tool",
        "Large_machinery",
    ]
    assert config["model"]["pair_feature_dim"] == 66
    assert config["model"]["hidden_dim"] == 128
    assert config["model"]["num_layers"] == 3
    assert config["training"]["batch_size"] == 4
    assert config["training"]["epochs"] == 30
    assert config["training"]["checkpoint_metric"] == "mR@5"
    assert config["loss"]["weights"] == {
        "relation": 1.0,
        "ppe": 0.2,
        "zone": 0.1,
        "interaction": 0.1,
        "legality": 0.1,
    }


def test_dense_pair_features_model_and_joint_loss():
    config = load_config(ROOT / "configs/paper_default.yaml")
    graph = load_graphs(ROOT / "examples/synthetic_graphs.json", config)[0]
    assert graph["pair_features"].shape == (6, 66)
    assert graph["pair_geometry"].shape == (6, 18)
    assert graph["spatial_evidence"].shape == (6, 8)

    model = RuleGroundedSGG(config, rule_grounded=True)
    model.eval()
    outputs = model(graph)
    assert outputs["relation_logits"].shape == (6, 5)
    assert outputs["rule_token"].shape == (6, 128)
    assert torch.allclose(
        outputs["assignments"][0]["probabilities"].sum(), torch.tensor(1.0)
    )
    losses = compute_joint_loss(outputs, graph, config, rule_grounded=True)
    assert set(losses) == {"relation", "ppe", "zone", "interaction", "legality", "total"}
    assert torch.isfinite(losses["total"])


def test_transformer_configuration_and_type_embeddings():
    config = load_config(ROOT / "configs/transformer.yaml")
    assert config["model"]["backbone"] == "transformer"
    assert config["model"]["num_heads"] == 4
    assert config["model"]["ffn_dim"] == 512

    baseline = RuleGroundedSGG(config, rule_grounded=False)
    grounded = RuleGroundedSGG(config, rule_grounded=True)
    assert isinstance(baseline.reasoner, TransformerReasoner)
    assert hasattr(baseline.reasoner, "subject_type_embedding")
    assert hasattr(baseline.reasoner, "object_type_embedding")
    assert baseline.object_encoder is None
    assert grounded.object_encoder is not None
    assert trainable_parameters(grounded) > trainable_parameters(baseline)
    assert baseline.rule_encoder is None
    assert baseline.assignment is None
    assert baseline.rule_classifier is None
    assert grounded.visual_classifier is None
