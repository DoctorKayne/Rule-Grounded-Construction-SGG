"""Joint objective from Methods Eqs. (21)--(25)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _ids(config: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    entities = {name: index for index, name in enumerate(config["classes"]["entities"])}
    predicates = {
        name: index for index, name in enumerate(config["classes"]["predicates"])
    }
    return entities, predicates


def _soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[0] == 0:
        return logits.sum() * 0.0
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def geometry_soft_targets(
    graph: dict[str, Any], config: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build geometry-based soft targets for zone and interaction relations."""
    entities, predicates = _ids(config)
    pair_index = graph["pair_index"]
    cats = graph["object_categories"]
    subject, obj = cats[pair_index[:, 0]], cats[pair_index[:, 1]]
    evidence, geometry = graph["spatial_evidence"], graph["pair_geometry"]
    thresholds = config["thresholds"]
    num_edges = pair_index.shape[0]
    num_predicates = len(predicates)
    zone_targets = evidence.new_zeros((num_edges, num_predicates))
    interaction_targets = evidence.new_zeros((num_edges, num_predicates))

    zone_mask = (subject == entities["Worker"]) & (obj == entities["Hazardous_area"])
    if zone_mask.any():
        foot_inside = geometry[:, 12]
        occupancy = evidence[:, 6]
        boundary = evidence[:, 5]
        iou = evidence[:, 4]
        region_overlap = geometry[:, 16]
        into_strength = torch.maximum(
            foot_inside,
            (occupancy / float(thresholds["zone_occupancy"])).clamp(0.0, 1.0),
        )
        near_evidence = torch.maximum(
            (1.0 - boundary / float(thresholds["near_boundary_distance"])).clamp(0.0, 1.0),
            torch.maximum(
                (iou / float(thresholds["zone_iou"])).clamp(0.0, 1.0),
                (region_overlap / float(thresholds["zone_overlap"])).clamp(0.0, 1.0),
            ),
        )
        near_strength = (1.0 - into_strength) * near_evidence
        normalizer = (into_strength + near_strength).clamp_min(1e-12)
        zone_targets[:, predicates["into"]] = into_strength / normalizer
        zone_targets[:, predicates["near"]] = near_strength / normalizer
        zone_mask &= (into_strength + near_strength) > 0.0

    interaction_mask = (subject == entities["Worker"]) & (
        (obj == entities["Tool"]) | (obj == entities["Large_machinery"])
    )
    if interaction_mask.any():
        contact = evidence[:, 7]
        boundary = evidence[:, 5]
        subject_inclusion = geometry[:, 16]
        machinery = obj == entities["Large_machinery"]
        performing_strength = (contact / float(thresholds["contact"])).clamp(0.0, 1.0)
        performing_strength = torch.where(
            machinery,
            torch.maximum(
                performing_strength,
                (subject_inclusion / float(thresholds["machinery_inclusion"])).clamp(
                    0.0, 1.0
                ),
            ),
            performing_strength,
        )
        near_strength = (1.0 - performing_strength) * (
            1.0 - boundary / float(thresholds["near_boundary_distance"])
        ).clamp(0.0, 1.0)
        normalizer = (performing_strength + near_strength).clamp_min(1e-12)
        interaction_targets[:, predicates["performing"]] = performing_strength / normalizer
        interaction_targets[:, predicates["near"]] = near_strength / normalizer
        interaction_mask &= (performing_strength + near_strength) > 0.0

    return zone_mask, zone_targets, interaction_mask, interaction_targets


def compute_joint_loss(
    outputs: dict[str, Any],
    graph: dict[str, Any],
    config: dict[str, Any],
    rule_grounded: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute ``Lrel + λppe Lppe + λzone Lzone + λint Lint + λlegal Llegal``."""
    logits = outputs["relation_logits"]
    weights = torch.tensor(
        config["loss"]["class_weights"], dtype=logits.dtype, device=logits.device
    )
    relation = F.cross_entropy(logits, graph["pair_labels"], weight=weights)
    zero = logits.sum() * 0.0
    parts = {
        "relation": relation,
        "ppe": zero,
        "zone": zero,
        "interaction": zero,
        "legality": zero,
    }
    if not rule_grounded:
        parts["total"] = relation
        return parts

    assignment_losses = []
    for item in outputs.get("assignments", []):
        logits_k = item["logits"].unsqueeze(0)
        target = torch.tensor(
            [int(item["target_position"])], dtype=torch.long, device=logits.device
        )
        assignment_losses.append(F.cross_entropy(logits_k, target))
    if assignment_losses:
        parts["ppe"] = torch.stack(assignment_losses).mean()

    zone_mask, zone_targets, interaction_mask, interaction_targets = geometry_soft_targets(
        graph, config
    )
    if zone_mask.any():
        parts["zone"] = _soft_cross_entropy(logits[zone_mask], zone_targets[zone_mask])
    if interaction_mask.any():
        parts["interaction"] = _soft_cross_entropy(
            logits[interaction_mask], interaction_targets[interaction_mask]
        )

    probabilities = torch.softmax(logits, dim=-1)
    parts["legality"] = (
        (1.0 - outputs["legality_mask"]) * probabilities
    ).sum(dim=-1).mean()

    lambdas = config["loss"]["weights"]
    parts["total"] = (
        parts["relation"]
        + float(lambdas["ppe"]) * parts["ppe"]
        + float(lambdas["zone"]) * parts["zone"]
        + float(lambdas["interaction"]) * parts["interaction"]
        + float(lambdas["legality"]) * parts["legality"]
    )
    return parts
