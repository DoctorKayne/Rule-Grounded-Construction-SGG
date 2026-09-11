"""Data loading and deterministic geometry construction."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration, resolving one optional relative ``extends``."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = config.pop("extends", None)
    if parent is None:
        return config
    return _deep_merge(load_config(path.parent / str(parent)), config)


def _box_terms(box: list[float], width: float, height: float) -> tuple[float, ...]:
    x1, y1, x2, y2 = (float(value) for value in box)
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid xyxy box {box} for image size {(width, height)}")
    w, h = x2 - x1, y2 - y1
    return x1, y1, x2, y2, w, h, (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _intersection(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def _contact_evidence(
    iou: float,
    overlap_s: float,
    overlap_o: float,
    boundary_distance: float,
    dx: float,
    dy: float,
    contact_gate: float,
    proximity_scale: float,
) -> float:
    """Compute normalized contact evidence from overlap, proximity, and layout."""
    local_overlap = max(iou, overlap_s, overlap_o)
    if proximity_scale <= 0.0 or contact_gate <= 0.0:
        return max(0.0, min(1.0, local_overlap))
    proximity = max(0.0, 1.0 - boundary_distance / proximity_scale)
    layout_distance = math.hypot(dx, dy)
    layout = max(0.0, 1.0 - layout_distance)
    proximity_layout = contact_gate * proximity * layout
    return max(0.0, min(1.0, max(local_overlap, proximity_layout)))


def geometry_and_evidence(
    subject_box: list[float],
    object_box: list[float],
    image_size: tuple[int, int],
    *,
    contact_gate: float = 0.02,
    proximity_scale: float = 0.20,
) -> tuple[list[float], list[float]]:
    """Return the 18-D pair geometry and the 8-D Eq. (4) evidence vector."""
    width, height = (float(image_size[0]), float(image_size[1]))
    s = _box_terms(subject_box, width, height)
    o = _box_terms(object_box, width, height)
    area_s, area_o = s[4] * s[5], o[4] * o[5]
    inter = _intersection(s, o)
    union = area_s + area_o - inter
    iou = inter / max(union, 1e-12)
    overlap_s = inter / max(area_s, 1e-12)
    overlap_o = inter / max(area_o, 1e-12)

    dx = (o[6] - s[6]) / width
    dy = (o[7] - s[7]) / height
    dw = math.log(max(o[4], 1e-12) / max(s[4], 1e-12))
    dh = math.log(max(o[5], 1e-12) / max(s[5], 1e-12))
    aspect_s = s[4] / max(s[5], 1e-12)
    aspect_o = o[4] / max(o[5], 1e-12)

    ux1, uy1 = min(s[0], o[0]) / width, min(s[1], o[1]) / height
    ux2, uy2 = max(s[2], o[2]) / width, max(s[3], o[3]) / height

    gap_x = max(o[0] - s[2], s[0] - o[2], 0.0)
    gap_y = max(o[1] - s[3], s[1] - o[3], 0.0)
    boundary_distance = math.hypot(gap_x, gap_y) / math.hypot(width, height)

    foot_x, foot_y = s[6], s[3]
    foot_inside = float(o[0] <= foot_x <= o[2] and o[1] <= foot_y <= o[3])
    lower = (s[0], s[1] + 2.0 * s[5] / 3.0, s[2], s[3])
    lower_area = max((lower[2] - lower[0]) * (lower[3] - lower[1]), 1e-12)
    occupancy = _intersection(lower, o) / lower_area

    contact = _contact_evidence(
        iou,
        overlap_s,
        overlap_o,
        boundary_distance,
        dx,
        dy,
        contact_gate,
        proximity_scale,
    )

    geometry = [
        dx,
        dy,
        math.log(max(area_o, 1e-12) / max(area_s, 1e-12)),
        aspect_s,
        aspect_o,
        iou,
        ux1,
        uy1,
        ux2,
        uy2,
        s[6] / width,
        s[7] / height,
        foot_inside,
        boundary_distance,
        occupancy,
        contact,
        overlap_s,
        overlap_o,
    ]
    evidence = [dx, dy, dw, dh, iou, boundary_distance, occupancy, contact]
    return geometry, evidence


def _vector(values: Any, size: int, name: str) -> list[float]:
    if not isinstance(values, list) or len(values) != size:
        raise ValueError(f"{name} must contain exactly {size} numbers")
    return [float(value) for value in values]


def scene_to_graph(scene: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Validate one JSON scene and convert it to the tensor graph contract."""
    width, height = int(scene["width"]), int(scene["height"])
    objects = scene.get("objects", [])
    if len(objects) < 2:
        raise ValueError("Each scene must contain at least two objects")

    entity_names = list(config["classes"]["entities"])
    predicate_names = list(config["classes"]["predicates"])
    entity_to_id = {name: index for index, name in enumerate(entity_names)}
    predicate_to_id = {name: index for index, name in enumerate(predicate_names)}

    categories, confidence, boxes, appearance = [], [], [], []
    for index, obj in enumerate(objects):
        category = str(obj["category"])
        if category not in entity_to_id:
            raise ValueError(f"Unknown object category {category!r} at index {index}")
        _box_terms(obj["box"], width, height)
        categories.append(entity_to_id[category])
        confidence.append(float(obj["confidence"]))
        boxes.append([float(value) for value in obj["box"]])
        appearance.append(_vector(obj["appearance"], 16, f"objects[{index}].appearance"))

    supplied: dict[tuple[int, int], dict[str, Any]] = {}
    for pair in scene.get("pairs", []):
        key = (int(pair["subject"]), int(pair["object"]))
        if key in supplied:
            raise ValueError(f"Duplicate ordered pair {key}")
        supplied[key] = pair
    expected = {(i, j) for i in range(len(objects)) for j in range(len(objects)) if i != j}
    if set(supplied) != expected:
        missing = sorted(expected - set(supplied))
        extra = sorted(set(supplied) - expected)
        raise ValueError(f"Dense ordered-pair contract violated; missing={missing}, extra={extra}")

    pair_index, union_features, geometries, evidence, labels = [], [], [], [], []
    contact_gate = float(config["thresholds"]["contact"])
    proximity_scale = float(config["thresholds"]["near_boundary_distance"])
    for subject, obj in sorted(expected):
        pair = supplied[(subject, obj)]
        predicate = str(pair["predicate"])
        if predicate not in predicate_to_id:
            raise ValueError(f"Unknown predicate {predicate!r} for pair {(subject, obj)}")
        geometry, semantic = geometry_and_evidence(
            boxes[subject],
            boxes[obj],
            (width, height),
            contact_gate=contact_gate,
            proximity_scale=proximity_scale,
        )
        pair_index.append([subject, obj])
        union_features.append(
            _vector(pair["union_appearance"], 16, f"pairs[{subject},{obj}].union_appearance")
        )
        geometries.append(geometry)
        evidence.append(semantic)
        labels.append(predicate_to_id[predicate])

    pair_index_t = torch.tensor(pair_index, dtype=torch.long)
    appearance_t = torch.tensor(appearance, dtype=torch.float32)
    geometry_t = torch.tensor(geometries, dtype=torch.float32)
    pair_features = torch.cat(
        [
            appearance_t[pair_index_t[:, 0]],
            appearance_t[pair_index_t[:, 1]],
            torch.tensor(union_features, dtype=torch.float32),
            geometry_t,
        ],
        dim=-1,
    )
    if pair_features.shape[-1] != 66:
        raise AssertionError(f"Expected 66-D pair features, got {pair_features.shape[-1]}")

    box_t = torch.tensor(boxes, dtype=torch.float32)
    object_geometry = torch.stack(
        [
            (box_t[:, 0] + box_t[:, 2]) / (2.0 * width),
            (box_t[:, 1] + box_t[:, 3]) / (2.0 * height),
            (box_t[:, 2] - box_t[:, 0]) / width,
            (box_t[:, 3] - box_t[:, 1]) / height,
        ],
        dim=-1,
    )

    assignments = []
    ppe_ids = {
        entity_to_id["Helmet"],
        entity_to_id["Safety_vest"],
        entity_to_id["Safety_belt"],
    }
    worker_id = entity_to_id["Worker"]
    for item in scene.get("ppe_assignments", []):
        ppe = int(item["ppe"])
        workers = [int(value) for value in item["candidate_workers"]]
        target = int(item["assigned_worker"])
        if ppe < 0 or ppe >= len(objects) or categories[ppe] not in ppe_ids:
            raise ValueError("ppe must index a Helmet, Safety_vest, or Safety_belt object")
        if not workers or any(
            worker < 0 or worker >= len(objects) or categories[worker] != worker_id
            for worker in workers
        ):
            raise ValueError("candidate_workers must index one or more Worker objects")
        if target not in workers:
            raise ValueError("assigned_worker must be one of candidate_workers")
        assignments.append(
            {"ppe": ppe, "workers": workers, "target_position": workers.index(target)}
        )

    return {
        "image_id": str(scene["image_id"]),
        "object_appearance": appearance_t,
        "object_categories": torch.tensor(categories, dtype=torch.long),
        "object_confidence": torch.tensor(confidence, dtype=torch.float32),
        "object_geometry": object_geometry,
        "pair_index": pair_index_t,
        "pair_features": pair_features,
        "pair_geometry": geometry_t,
        "spatial_evidence": torch.tensor(evidence, dtype=torch.float32),
        "pair_labels": torch.tensor(labels, dtype=torch.long),
        "ppe_assignments": assignments,
    }


def load_graphs(path: str | Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scenes = payload if isinstance(payload, list) else payload.get("scenes", [])
    if not scenes:
        raise ValueError("The dataset JSON contains no scenes")
    return [scene_to_graph(scene, config) for scene in scenes]


def graph_to_device(graph: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in graph.items()
    }
