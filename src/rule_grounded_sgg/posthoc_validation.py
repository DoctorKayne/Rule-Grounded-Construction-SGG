"""Inference-stage validation and worker-centered aggregation (Sec. 3.5)."""

from __future__ import annotations

from typing import Any

import torch

from .rule_encoder import build_legality_table


def _ids(config: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    entities = {name: index for index, name in enumerate(config["classes"]["entities"])}
    predicates = {
        name: index for index, name in enumerate(config["classes"]["predicates"])
    }
    return entities, predicates


def relation_validity(
    predictions: torch.Tensor,
    graph: dict[str, Any],
    config: dict[str, Any],
    probabilities: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one structural-validity flag per predicted edge."""
    entities, predicates = _ids(config)
    pair_index = graph["pair_index"]
    categories = graph["object_categories"]
    subject = categories[pair_index[:, 0]]
    obj = categories[pair_index[:, 1]]
    geometry = graph["pair_geometry"]
    evidence = graph["spatial_evidence"]
    thresholds = config["thresholds"]
    legality = build_legality_table(config).to(predictions.device)[subject, obj]
    valid = legality.gather(1, predictions[:, None]).squeeze(1).bool()

    into = predictions == predicates["into"]
    into_supported = (geometry[:, 12] > 0.5) | (
        evidence[:, 6] >= float(thresholds["zone_occupancy"])
    )
    valid &= ~into | into_supported

    performing = predictions == predicates["performing"]
    performing_supported = evidence[:, 7] >= float(thresholds["contact"])
    machinery = obj == entities["Large_machinery"]
    performing_supported |= machinery & (
        geometry[:, 16] >= float(thresholds["machinery_inclusion"])
    )
    valid &= ~performing | performing_supported

    near = predictions == predicates["near"]
    zone_pair = (subject == entities["Worker"]) & (obj == entities["Hazardous_area"])
    zone_near_supported = (
        (evidence[:, 5] < float(thresholds["near_boundary_distance"]))
        | (evidence[:, 4] > float(thresholds["zone_iou"]))
        | (geometry[:, 16] > float(thresholds["zone_overlap"]))
    ) & ~into_supported
    valid &= ~(near & zone_pair) | zone_near_supported

    interaction_pair = (subject == entities["Worker"]) & (
        (obj == entities["Tool"]) | (obj == entities["Large_machinery"])
    )
    layout_distance = torch.sqrt(evidence[:, 0].square() + evidence[:, 1].square())
    interaction_near_supported = (
        (evidence[:, 5] < float(thresholds["near_boundary_distance"]))
        | (layout_distance < float(thresholds["near_boundary_distance"]))
    ) & ~performing_supported
    valid &= ~(near & interaction_pair) | interaction_near_supported

    wear = predictions == predicates["wear"]
    ppe_ids = {entities["Helmet"], entities["Safety_vest"], entities["Safety_belt"]}
    for ppe_index in pair_index[wear, 1].unique().tolist():
        edge_ids = torch.where(wear & (pair_index[:, 1] == int(ppe_index)))[0]
        if edge_ids.numel() <= 1 or int(categories[int(ppe_index)]) not in ppe_ids:
            continue
        if probabilities is None:
            keep = edge_ids[0]
        else:
            keep = edge_ids[
                probabilities[edge_ids, predicates["wear"]].argmax()
            ]
        keep_was_valid = valid[keep].clone()
        valid[edge_ids] = False
        valid[keep] = keep_was_valid
    return valid


def apply_posthoc_validation(
    predictions: torch.Tensor,
    graph: dict[str, Any],
    config: dict[str, Any],
    probabilities: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the limited checks specified in Methods Sec. 3.5.1."""
    del probabilities
    entities, predicates = _ids(config)
    pair_index = graph["pair_index"]
    categories = graph["object_categories"]
    subject = categories[pair_index[:, 0]]
    obj = categories[pair_index[:, 1]]
    geometry = graph["pair_geometry"]
    evidence = graph["spatial_evidence"]
    thresholds = config["thresholds"]
    legality = build_legality_table(config).to(predictions.device)[subject, obj]
    valid = legality.gather(1, predictions[:, None]).squeeze(1).bool()

    into = predictions == predicates["into"]
    into_supported = (geometry[:, 12] > 0.5) | (
        evidence[:, 6] >= float(thresholds["zone_occupancy"])
    )
    valid &= ~into | into_supported

    performing = predictions == predicates["performing"]
    performing_supported = evidence[:, 7] >= float(thresholds["contact"])
    performing_supported |= (obj == entities["Large_machinery"]) & (
        geometry[:, 16] >= float(thresholds["machinery_inclusion"])
    )
    valid &= ~performing | performing_supported

    foreground = predictions != predicates["none"]
    keep = valid | ~foreground
    validated = predictions.clone()
    validated[~keep] = predicates["none"]
    return validated, keep


def aggregate_worker_relations(
    predictions: torch.Tensor,
    graph: dict[str, Any],
    config: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Organize predicted relations into worker-centered safety evidence.

    Relevant edges are grouped into PPE, hazardous-area, and tool/machinery
    relations, and each group is summarized by its strongest predicted state.
    """
    entities, predicates = _ids(config)
    predicate_names = list(config["classes"]["predicates"])
    pair_index = graph["pair_index"]
    categories = graph["object_categories"]
    ppe_ids = {entities["Helmet"], entities["Safety_vest"], entities["Safety_belt"]}
    interaction_ids = {entities["Tool"], entities["Large_machinery"]}

    result: dict[int, dict[str, Any]] = {}
    for worker in torch.where(categories == entities["Worker"])[0].tolist():
        result[int(worker)] = {
            "ppe": [],
            "zone": [],
            "interaction": [],
            "states": {"ppe": "none", "zone": "none", "interaction": "none"},
        }

    for edge, (subject, obj) in enumerate(pair_index.tolist()):
        if subject not in result:
            continue
        predicate_id = int(predictions[edge])
        predicate = predicate_names[predicate_id]
        if predicate == "none":
            continue
        category = int(categories[obj])
        if category in ppe_ids:
            result[subject]["ppe"].append(edge)
        elif category == entities["Hazardous_area"]:
            result[subject]["zone"].append(edge)
        elif category in interaction_ids:
            result[subject]["interaction"].append(edge)

    for worker, item in result.items():
        del worker
        if any(int(predictions[edge]) == predicates["wear"] for edge in item["ppe"]):
            item["states"]["ppe"] = "wear"

        zone_predicates = {int(predictions[edge]) for edge in item["zone"]}
        if predicates["into"] in zone_predicates:
            item["states"]["zone"] = "into"
        elif predicates["near"] in zone_predicates:
            item["states"]["zone"] = "near"

        interaction_predicates = {int(predictions[edge]) for edge in item["interaction"]}
        if predicates["performing"] in interaction_predicates:
            item["states"]["interaction"] = "performing"
        elif predicates["near"] in interaction_predicates:
            item["states"]["interaction"] = "near"

    return result
