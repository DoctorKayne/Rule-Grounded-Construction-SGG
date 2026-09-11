"""MP-GNN and Transformer reference models for rule-grounded relation learning."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .assignment import PPEAssignment
from .rule_encoder import RuleEncoder, build_legality_table


def _mlp(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
        nn.LayerNorm(output_dim),
        nn.Dropout(dropout),
        nn.Linear(output_dim, output_dim),
    )


class ObjectEncoder(nn.Module):
    """Eq. (12): appearance, category, confidence, and normalized box geometry."""

    def __init__(self, num_entities: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.category_embedding = nn.Embedding(num_entities, 16)
        self.encoder = nn.Sequential(
            nn.Linear(16 + 16 + 1 + 4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, graph: dict[str, Any]) -> torch.Tensor:
        return self.encoder(
            torch.cat(
                [
                    graph["object_appearance"],
                    self.category_embedding(graph["object_categories"]),
                    graph["object_confidence"].unsqueeze(-1),
                    graph["object_geometry"],
                ],
                dim=-1,
            )
        )


class MessagePassingBlock(nn.Module):
    """Update directed pairs, mean-pool incident pairs, then update objects."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.pair_update = _mlp(3 * hidden_dim, hidden_dim, dropout)
        self.object_update = _mlp(2 * hidden_dim, hidden_dim, dropout)
        self.pair_norm = nn.LayerNorm(hidden_dim)
        self.object_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        object_states: torch.Tensor,
        pair_states: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        subject, obj = pair_index[:, 0], pair_index[:, 1]
        pair_delta = self.pair_update(
            torch.cat([pair_states, object_states[subject], object_states[obj]], dim=-1)
        )
        pair_states = self.pair_norm(pair_states + pair_delta)

        messages = object_states.new_zeros(object_states.shape)
        counts = object_states.new_zeros((object_states.shape[0], 1))
        messages.index_add_(0, subject, pair_states)
        messages.index_add_(0, obj, pair_states)
        ones = object_states.new_ones((pair_states.shape[0], 1))
        counts.index_add_(0, subject, ones)
        counts.index_add_(0, obj, ones)
        messages = messages / counts.clamp_min(1.0)
        object_delta = self.object_update(torch.cat([object_states, messages], dim=-1))
        object_states = self.object_norm(object_states + object_delta)
        return object_states, pair_states


class MessagePassingReasoner(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MessagePassingBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        object_states: torch.Tensor,
        pair_states: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            object_states, pair_states = block(object_states, pair_states, pair_index)
        return object_states, pair_states


class TransformerReasoner(nn.Module):
    """Three pre-norm relation-token blocks used for the backbone comparison."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        num_entities: int,
    ):
        super().__init__()
        self.subject_type_embedding = nn.Embedding(num_entities, hidden_dim)
        self.object_type_embedding = nn.Embedding(num_entities, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        pair_states: torch.Tensor,
        pair_index: torch.Tensor,
        object_categories: torch.Tensor,
    ) -> torch.Tensor:
        subject, obj = pair_index[:, 0], pair_index[:, 1]
        relation_tokens = (
            pair_states
            + self.subject_type_embedding(object_categories[subject])
            + self.object_type_embedding(object_categories[obj])
        )
        return self.encoder(relation_tokens.unsqueeze(0)).squeeze(0)


class RuleGroundedSGG(nn.Module):
    """Reference implementation of Eqs. (12)--(25)."""

    def __init__(self, config: dict[str, Any], rule_grounded: bool = True):
        super().__init__()
        model_cfg = config["model"]
        hidden_dim = int(model_cfg["hidden_dim"])
        dropout = float(model_cfg["dropout"])
        num_entities = len(config["classes"]["entities"])
        num_predicates = len(config["classes"]["predicates"])
        self.rule_grounded = bool(rule_grounded)
        self.config = config
        self.backbone = str(model_cfg["backbone"])

        # Object states are used by the MP-GNN backbone and PPE assignment.
        needs_object_encoder = self.backbone == "mpgnn" or self.rule_grounded
        self.object_encoder: ObjectEncoder | None = (
            ObjectEncoder(num_entities, hidden_dim, dropout) if needs_object_encoder else None
        )

        self.pair_encoder = nn.Sequential(
            nn.Linear(int(model_cfg["pair_feature_dim"]), hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        if self.backbone == "mpgnn":
            self.reasoner: nn.Module = MessagePassingReasoner(
                hidden_dim, int(model_cfg["num_layers"]), dropout
            )
        elif self.backbone == "transformer":
            self.reasoner = TransformerReasoner(
                hidden_dim,
                int(model_cfg["num_layers"]),
                int(model_cfg["num_heads"]),
                int(model_cfg["ffn_dim"]),
                dropout,
                num_entities,
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        if self.rule_grounded:
            self.rule_encoder: RuleEncoder | None = RuleEncoder(
                num_entities=num_entities,
                num_predicates=num_predicates,
                evidence_dim=int(model_cfg["spatial_evidence_dim"]),
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            self.assignment: PPEAssignment | None = PPEAssignment(
                hidden_dim, int(model_cfg["spatial_evidence_dim"]), dropout
            )
            self.rule_classifier: nn.Linear | None = nn.Linear(2 * hidden_dim, num_predicates)
            self.visual_classifier: nn.Linear | None = None
        else:
            self.rule_encoder = None
            self.assignment = None
            self.rule_classifier = None
            self.visual_classifier = nn.Linear(hidden_dim, num_predicates)

        self.register_buffer("legality_table", build_legality_table(config), persistent=True)

    def forward(self, graph: dict[str, Any]) -> dict[str, Any]:
        pair_features = graph["pair_features"]
        if pair_features.ndim != 2 or pair_features.shape[-1] != 66:
            raise ValueError("pair_features must have shape (E, 66)")
        pair_states = self.pair_encoder(pair_features)

        if isinstance(self.reasoner, TransformerReasoner):
            pair_states = self.reasoner(
                pair_states,
                graph["pair_index"],
                graph["object_categories"],
            )
            object_states = (
                self.object_encoder(graph) if self.object_encoder is not None else None
            )
        else:
            assert self.object_encoder is not None
            object_states = self.object_encoder(graph)
            object_states, pair_states = self.reasoner(
                object_states, pair_states, graph["pair_index"]
            )

        subject = graph["object_categories"][graph["pair_index"][:, 0]]
        obj = graph["object_categories"][graph["pair_index"][:, 1]]
        legality = self.legality_table[subject, obj]
        if self.rule_grounded:
            assert self.rule_encoder is not None
            assert self.assignment is not None
            assert self.rule_classifier is not None
            assert object_states is not None
            rule_token = self.rule_encoder(subject, obj, legality, graph["spatial_evidence"])
            logits = self.rule_classifier(torch.cat([pair_states, rule_token], dim=-1))
            assignments = self.assignment(
                object_states,
                graph["pair_index"],
                graph["spatial_evidence"],
                graph.get("ppe_assignments", []),
            )
        else:
            assert self.visual_classifier is not None
            rule_token = None
            logits = self.visual_classifier(pair_states)
            assignments = []
        return {
            "relation_logits": logits,
            "legality_mask": legality,
            "graph_context": pair_states,
            "rule_token": rule_token,
            "assignments": assignments,
        }
