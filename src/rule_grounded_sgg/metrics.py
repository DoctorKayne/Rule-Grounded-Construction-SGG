"""Label-based and structural-compliance metrics used by the paper."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import torch

from .posthoc_validation import apply_posthoc_validation, relation_validity


def foreground_prf(
    labels: torch.Tensor, predictions: torch.Tensor, none_id: int
) -> dict[str, float]:
    """Micro precision, recall, and F1 over non-``none`` safety relations."""
    true_fg = labels != none_id
    pred_fg = predictions != none_id
    exact = labels == predictions
    tp = int((true_fg & pred_fg & exact).sum())
    fp = int((pred_fg & ~(true_fg & exact)).sum())
    fn = int((true_fg & ~(pred_fg & exact)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "overall_f1": f1}


def _foreground_candidates(
    probabilities: torch.Tensor, foreground_ids: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    foreground_probabilities = probabilities[:, foreground_ids]
    confidence, local_prediction = foreground_probabilities.max(dim=-1)
    prediction = torch.tensor(foreground_ids, device=probabilities.device)[local_prediction]
    return confidence, prediction


def _topk_counts(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    foreground_ids: list[int],
    topk: Iterable[int],
    candidate_predictions: torch.Tensor | None = None,
    candidate_keep: torch.Tensor | None = None,
) -> tuple[dict[int, int], dict[int, dict[int, int]], dict[int, int]]:
    confidence, ranked_prediction = _foreground_candidates(probabilities, foreground_ids)
    if candidate_predictions is not None:
        ranked_prediction = candidate_predictions
        confidence = probabilities.gather(1, ranked_prediction[:, None]).squeeze(1)
    eligible = torch.ones_like(ranked_prediction, dtype=torch.bool)
    if candidate_keep is not None:
        eligible &= candidate_keep
    foreground_tensor = torch.tensor(foreground_ids, device=labels.device)
    eligible &= (ranked_prediction[:, None] == foreground_tensor[None, :]).any(dim=1)

    ranked_confidence = confidence.masked_fill(~eligible, float("-inf"))
    order = ranked_confidence.argsort(descending=True)
    available = int(eligible.sum())
    total_by_class = {class_id: int((labels == class_id).sum()) for class_id in foreground_ids}
    correct_total: dict[int, int] = {}
    correct_by_class: dict[int, dict[int, int]] = {}
    for k in topk:
        selected = order[: min(int(k), available)]
        correct = ranked_prediction[selected] == labels[selected]
        selected_labels = labels[selected]
        correct_total[int(k)] = int(correct.sum())
        correct_by_class[int(k)] = {
            class_id: int((correct & (selected_labels == class_id)).sum())
            for class_id in foreground_ids
        }
    return correct_total, correct_by_class, total_by_class


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    graphs: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    posthoc: bool = False,
) -> dict[str, float]:
    """Evaluate image graphs with within-image top-K ranking."""
    model.eval()
    predicate_names = list(config["classes"]["predicates"])
    none_id = predicate_names.index("none")
    foreground_ids = [index for index in range(len(predicate_names)) if index != none_id]
    topk = [int(value) for value in config["evaluation"]["topk"]]
    labels_all, predictions_all = [], []
    topk_correct = defaultdict(int)
    topk_by_class = {k: defaultdict(int) for k in topk}
    support_by_class = defaultdict(int)
    valid_foreground = 0
    predicted_foreground = 0

    from .data import graph_to_device

    for original_graph in graphs:
        graph = graph_to_device(original_graph, device)
        outputs = model(graph)
        probabilities = torch.softmax(outputs["relation_logits"], dim=-1)
        predictions = probabilities.argmax(dim=-1)
        if posthoc:
            predictions, _ = apply_posthoc_validation(
                predictions, graph, config, probabilities
            )
        labels = graph["pair_labels"]
        labels_all.append(labels.cpu())
        predictions_all.append(predictions.cpu())

        candidate_predictions = None
        candidate_keep = None
        if posthoc:
            _, raw_foreground_predictions = _foreground_candidates(
                probabilities, foreground_ids
            )
            candidate_predictions, candidate_keep = apply_posthoc_validation(
                raw_foreground_predictions, graph, config, probabilities
            )

        totals, by_class, supports = _topk_counts(
            labels,
            probabilities,
            foreground_ids,
            topk,
            candidate_predictions=candidate_predictions,
            candidate_keep=candidate_keep,
        )
        for k, value in totals.items():
            topk_correct[k] += value
        for k, values in by_class.items():
            for class_id, value in values.items():
                topk_by_class[k][class_id] += value
        for class_id, value in supports.items():
            support_by_class[class_id] += value

        foreground = predictions != none_id
        validity = relation_validity(predictions, graph, config, probabilities)
        predicted_foreground += int(foreground.sum())
        valid_foreground += int((foreground & validity).sum())

    labels_t = torch.cat(labels_all)
    predictions_t = torch.cat(predictions_all)
    metrics = foreground_prf(labels_t, predictions_t, none_id)
    total_foreground = sum(support_by_class.values())
    for k in topk:
        metrics[f"R@{k}"] = (
            100.0 * topk_correct[k] / total_foreground if total_foreground else 0.0
        )
        recalls = [
            topk_by_class[k][class_id] / support_by_class[class_id]
            for class_id in foreground_ids
            if support_by_class[class_id] > 0
        ]
        metrics[f"mR@{k}"] = 100.0 * sum(recalls) / len(recalls) if recalls else 0.0
    scr = 100.0 * valid_foreground / predicted_foreground if predicted_foreground else 100.0
    metrics["SCR"] = scr
    metrics["RVR"] = 100.0 - scr
    metrics["predicted_foreground"] = float(predicted_foreground)
    return metrics
