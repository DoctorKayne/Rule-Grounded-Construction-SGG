"""Evaluate a reference checkpoint with F1, R@K, mR@K, SCR, and RVR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rule_grounded_sgg.data import load_config, load_graphs  # noqa: E402
from rule_grounded_sgg.metrics import evaluate  # noqa: E402
from rule_grounded_sgg.model import RuleGroundedSGG  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--posthoc", action="store_true", help="Evaluate the + Validate variant")
    parser.add_argument("--output", help="Optional aggregate JSON output")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mode = str(checkpoint.get("mode", "rule_grounded"))
    model = RuleGroundedSGG(config, rule_grounded=mode == "rule_grounded")
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = evaluate(model, load_graphs(args.data, config), config, device, args.posthoc)
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
