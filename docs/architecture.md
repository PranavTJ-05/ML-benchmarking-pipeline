# Benchmark architecture decisions

This document records the agreed architecture. The YAML files under `configs/`
are the machine-readable source for preprocessing and regression behavior.

## Reproducibility

- Dataset: Google Speech Commands v2 (v0.02), official train/validation/test split.
- Labels: ten default keywords plus `unknown` and generated `silence`.
- MFCCs are computed outside the model; training-set mean/std are frozen into
  each resolved run configuration.
- Every run records seed, git commit, config path, split, runtime, device,
  threads, timestamp, and artifact SHA-256.
- JSON is the source of truth; Markdown, CSV, and plots are derived artifacts.

## Devices and latency

Development and CI use x86 Linux. Final ARM measurements use a physical
Raspberry Pi, or an Android phone with TFLite benchmark tooling over adb. The
exact Android SoC is recorded. QEMU/emulated ARM is never reported as device
latency. Model-only single-item latency is the headline; MFCC preprocessing is
measured separately.

## Quantization and conversion

Use static post-training INT8 calibrated from a few hundred training clips only.
Consider QAT only after measured accuracy loss exceeds one percentage point.
Validate AI Edge Torch or onnx2tf early because TFLite conversion is the riskiest
integration step.
