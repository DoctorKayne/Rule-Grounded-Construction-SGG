"""Train the Baseline or Rule-Grounded reference model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rule_grounded_sgg.data import graph_to_device, load_config, load_graphs  # noqa: E402
from rule_grounded_sgg.losses import compute_joint_loss  # noqa: E402
from rule_grounded_sgg.metrics import evaluate  # noqa: E402
from rule_grounded_sgg.model import RuleGroundedSGG  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batches(items: list, batch_size: int, rng: random.Random):
    order = list(range(len(items)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [items[index] for index in order[start : start + batch_size]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", required=True, help="Training graph JSON")
    parser.add_argument("--val", required=True, help="Validation graph JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("baseline", "rule_grounded"), default="rule_grounded")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, help="Optional smoke-test override")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["training"]["seed"])
    epochs = int(args.epochs if args.epochs is not None else config["training"]["epochs"])
    rule_grounded = args.mode == "rule_grounded"
    set_seed(seed)
    rng = random.Random(seed)

    train_graphs = load_graphs(args.train, config)
    val_graphs = load_graphs(args.val, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RuleGroundedSGG(config, rule_grounded=rule_grounded).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(config["training"]["learning_rate"])
        * float(config["training"]["minimum_lr_ratio"]),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = float("-inf")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch in batches(train_graphs, int(config["training"]["batch_size"]), rng):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for original_graph in batch:
                graph = graph_to_device(original_graph, device)
                parts = compute_joint_loss(
                    model(graph), graph, config, rule_grounded=rule_grounded
                )
                losses.append(parts["total"])
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip"])
            )
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        scheduler.step()

        metrics = evaluate(model, val_graphs, config, device, posthoc=False)
        record = {
            "epoch": epoch,
            "training_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
            "validation": metrics,
        }
        history.append(record)
        score = float(metrics["mR@5"])
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mode": args.mode,
                    "seed": seed,
                    "epoch": epoch,
                    "config": config,
                    "validation": metrics,
                },
                output_dir / "best.pt",
            )
        print(json.dumps(record, sort_keys=True))

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
