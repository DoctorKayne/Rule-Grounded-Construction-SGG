"""Differentiable PPE-to-worker assignment from Methods Sec. 3.4.2."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PPEAssignment(nn.Module):
    """Score each PPE item against its candidate workers.

    The normalization is a Softmax over candidate workers for each PPE item.
    Consequently, every returned probability vector sums to one and is trained
    against the annotated worker index with cross-entropy.
    """

    def __init__(self, hidden_dim: int = 128, evidence_dim: int = 8, dropout: float = 0.3):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim + evidence_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        object_states: torch.Tensor,
        pair_index: torch.Tensor,
        spatial_evidence: torch.Tensor,
        assignments: list[dict[str, Any]],
    ) -> list[dict[str, torch.Tensor | int]]:
        edge_lookup = {
            (int(subject), int(obj)): index
            for index, (subject, obj) in enumerate(pair_index.tolist())
        }
        outputs: list[dict[str, torch.Tensor | int]] = []
        for item in assignments:
            ppe = int(item["ppe"])
            workers = [int(value) for value in item["workers"]]
            if not workers:
                continue
            evidence = torch.stack(
                [spatial_evidence[edge_lookup[(worker, ppe)]] for worker in workers]
            )
            ppe_state = object_states[ppe].unsqueeze(0).expand(len(workers), -1)
            worker_states = object_states[torch.tensor(workers, device=object_states.device)]
            logits = self.scorer(torch.cat([ppe_state, worker_states, evidence], dim=-1)).squeeze(-1)
            outputs.append(
                {
                    "logits": logits,
                    "probabilities": torch.softmax(logits, dim=0),
                    "target_position": int(item["target_position"]),
                }
            )
        return outputs
