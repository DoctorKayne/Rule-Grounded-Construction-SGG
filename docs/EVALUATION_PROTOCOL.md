# Evaluation protocol

The reference implementation reports label-based recognition metrics and structural-compliance metrics.

## Label-based recognition

- **Overall F1** is micro F1 over the four foreground predicates (`wear`, `into`, `near`, and `performing`); `none` is excluded as a target class.
- **R@K** ranks the best foreground prediction for each ordered pair within each image and measures recovery of foreground ground-truth relations in the top K.
- **mR@K** is the arithmetic mean of predicate-wise Recall@K over foreground predicates represented in the evaluated set.

## Structural compliance

- **SCR** is the percentage of predicted foreground edges satisfying the configured object-predicate legality, PPE uniqueness, zone-boundary, and interaction-evidence checks.
- **RVR** is `100 - SCR`.

SCR and RVR quantify structural consistency and are reported alongside F1, R@K, and mR@K.

## Primary and validated inference

Primary forward results are evaluated without post-hoc validation. The `+ Validate` variants use `--posthoc`, which removes illegal category-predicate combinations and unsupported `into` or `performing` predictions after classification. Top-K metrics are then computed from the validated candidate set.

The checkpoint selected during training is the epoch with the highest validation mR@5. The repeated-run protocol uses seeds 41, 42, and 43.
