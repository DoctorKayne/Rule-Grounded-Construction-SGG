# Data format

Input data are represented as precomputed image graphs. Construction-site images and annotations are not included.

Each JSON file contains a list of scenes. A scene has:

- `image_id`: an arbitrary identifier;
- `width`, `height`: image dimensions used to normalize box geometry;
- `objects`: detector boxes, categories, confidence values, and one 16-D ROI appearance vector per object;
- `pairs`: every ordered pair `(i, j)` with `i != j`, including a 16-D union-region ROI appearance vector and a predicate label;
- `ppe_assignments`: optional annotated PPE-to-worker assignments used by the Rule-Grounded training objective.

Boxes use pixel-space `xyxy` coordinates. Categories and predicates must match `configs/paper_default.yaml`.

The loader constructs the 66-D pair feature in this order:

1. subject ROI appearance: 16 dimensions;
2. object ROI appearance: 16 dimensions;
3. union-region ROI appearance: 16 dimensions;
4. pairwise geometry: 18 dimensions.

The appearance vectors are expected to be ROI-pooled descriptors from the feature map of the frozen detector.

The 8-D spatial-semantic evidence follows Eq. (4): relative position (2), scale difference (2), IoU, normalized boundary distance, lower-body occupancy, and normalized contact evidence. Thresholds are used for auxiliary targets and validation.

`examples/synthetic_graphs.json` uses arbitrary coordinates and feature values for interface testing.
