# Phase 2: CPU baseline

Install with `pip install -e '.[training,dev]'`. Run from the project root:

```bash
python -m kws_benchmark.training.train --config configs/train.yaml
python -m kws_benchmark.training.evaluate --checkpoint artifacts/training/baseline.pt
pytest -q
```

Set dataset and manifest locations in `configs/train.yaml`. Inputs are extracted
once into memory. MFCCs have shape `(N, 1, 101, 40)`; normalization is fitted only
on training features and stored in `resolved.yaml` and each checkpoint.
Large datasets may require substantial RAM; streaming/caching is future work.

Defaults: CPU, one thread, seed 1337, Adam at 0.001, batch 64, 30 epochs,
five non-improving validation epochs of patience. Two 64-channel candidates
with three/four depthwise-separable blocks are compared. Selection uses validation
accuracy exclusively; evaluation uses the frozen baseline and test split.
JSON reports include confusion matrices, per-class precision/recall/F1, parameter
count, checkpoint bytes, config, dataset version, manifest hash and commit (null
outside Git). Checkpoints include weights/BatchNorm state, not optimizer resume state.

## Current limitations and phase gates

Phase 1 provides basic MFCCs, WAV reading, official-list manifest discovery and
unknown downsampling. Its silence entry is still a shared background-noise
placeholder, not isolated generated clips. Training rejects overlapping split
paths, so this must be completed before a real-data baseline run. YAML class
names are quoted to preserve labels such as yes/no/on/off as strings.
The synthetic training test demonstrates reproducibility/serialization only.
No Speech Commands baseline accuracy or final checkpoint is claimed.

The Phase 2 accuracy target is 0.90 (90%). The full dataset has not been
downloaded or trained here, so the real-data accuracy gate remains pending.
