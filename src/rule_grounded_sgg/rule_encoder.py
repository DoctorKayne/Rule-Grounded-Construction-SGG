"""Object-predicate legality and spatial-semantic rule encoding."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def build_legality_table(config: dict[str, Any]) -> torch.Tensor:
    entities = list(config["classes"]["entities"])
    predicates = list(config["classes"]["predicates"])
    entity_to_id = {name: index for index, name in enumerate(entities)}
    predicate_to_id = {name: index for index, name in enumerate(predicates)}
    none_id = predicate_to_id["none"]

    # Dense ordered pairs are retained.  Unlisted category combinations may
    # predict only ``none`` rather than being removed from the graph.
    table = torch.zeros(len(entities), len(entities), len(predicates), dtype=torch.float32)
    table[:, :, none_id] = 1.0
    for rule in config["rules"]["object_predicate"]:
        subject = entity_to_id[rule["subject"]]
        obj = entity_to_id[rule["object"]]
        table[subject, obj] = 0.0
        for predicate in rule["allow"]:
            table[subject, obj, predicate_to_id[predicate]] = 1.0
    return table


class RuleEncoder(nn.Module):
    """Encode categories, a 5-D legality mask, and 8-D evidence to 128-D."""

    def __init__(
        self,
        num_entities: int,
        num_predicates: int = 5,
        evidence_dim: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.subject_embedding = nn.Embedding(num_entities, hidden_dim)
        self.object_embedding = nn.Embedding(num_entities, hidden_dim)
        self.legality_projection = nn.Linear(num_predicates, hidden_dim)
        self.evidence_projection = nn.Linear(evidence_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        subject_categories: torch.Tensor,
        object_categories: torch.Tensor,
        legality_mask: torch.Tensor,
        spatial_evidence: torch.Tensor,
    ) -> torch.Tensor:
        return self.fusion(
            torch.cat(
                [
                    self.subject_embedding(subject_categories),
                    self.object_embedding(object_categories),
                    self.legality_projection(legality_mask),
                    self.evidence_projection(spatial_evidence),
                ],
                dim=-1,
            )
        )
