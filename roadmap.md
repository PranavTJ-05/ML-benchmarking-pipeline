# Edge Keyword Spotter - Project Roadmap

## Vision

Build a compact keyword-spotting model and a reproducible benchmarking
pipeline that compares PyTorch, ONNX Runtime, and TFLite on x86 Linux and one
physical ARM device.

The project is complete when a clean environment can reproduce the model
artifacts, accuracy results, latency distributions, memory measurements, and
CI regression decisions from configuration files.

## Five execution phases

Implementation status: Phase 1 has basic feature extraction and split manifests;
isolated silence generation and complete input validation remain open. Phase 2
has CPU training, validation-selected candidate search, checkpoint evaluation and
synthetic reproducibility tests. Real-data training and the accuracy gate remain
pending; see `docs/training.md` for commands and limitations.

### Phase 1 - Data and preprocessing

**Goal:** establish a deterministic MFCC input contract.

Tasks:

- use Google Speech Commands v2 (v0.02), its official split, and the default
  ten keywords plus `unknown` and generated `silence`;
- define 16 kHz, 1-second clips, 40 MFCCs, n_fft=512, 30 ms windows, 10 ms
  hops, and training-set-only mean/std normalization;
- implement MFCC extraction and normalization;
- create a small fixture dataset for fast tests;
- test shape, determinism, labels, corrupt inputs, and split isolation.

Gate:

- identical inputs produce identical features;
- all runtimes can consume the documented feature shape;
- train/validation/test splits are reproducible and non-overlapping.

### Phase 2 - PyTorch baseline

**Goal:** train and evaluate a compact depthwise-separable CNN.

Tasks:

- implement the configurable DS-CNN;
- add seeded training, checkpoints, early stopping, and evaluation;
- report accuracy, confusion matrix, parameter count, and model size;
- run a configuration-driven search over `K` model configurations;
- select and freeze a baseline checkpoint.

Gate:

- baseline accuracy target is met;
- training and evaluation are reproducible;
- model serialization and inference tests pass;
- experiment metadata is saved with every result.

### Phase 3 - Export and quantization

**Goal:** produce validated FP32 and INT8 ONNX/TFLite artifacts.

Tasks:

- export PyTorch to ONNX and validate the graph;
- implement ONNX Runtime inference;
- convert to TFLite;
- create representative calibration data;
- produce FP32 and INT8 artifacts;
- compare predictions and accuracy across formats;
- record size and quantization accuracy loss.

Gate:

- all artifacts load in a clean environment;
- prediction agreement and numerical-error thresholds pass;
- quantization accuracy loss is measured, not assumed;
- exported artifact checksums are recorded.

### Phase 4 - Benchmark harness

**Goal:** measure model/runtime/device combinations consistently.

Tasks:

- implement a config-driven benchmark CLI;
- add PyTorch, ONNX Runtime, and TFLite runners;
- measure warmup, p50, p95, mean, standard deviation, memory, and size;
- record runtime/device/environment metadata;
- benchmark x86 and one physical ARM device; never report QEMU/emulated ARM;
- make JSON the source of truth and generate CSV, Markdown, and plots from it;
- report model-only latency as the headline metric and preprocessing separately.

Gate:

- at least `M` combinations are measured;
- every result is traceable to a config and artifact checksum;
- smoke and final benchmarks are clearly separated;
- results can be compared across commits.

### Phase 5 - CI and release-quality documentation

**Goal:** prevent regressions and make the project portfolio-ready.

Tasks:

- add GitLab CI stages for lint, tests, smoke training, export, benchmark,
  regression, and artifact packaging;
- add accuracy, p95 latency, model-size, and compatibility gates from
  `configs/regression_thresholds.yaml`;
- publish reports and model artifacts from CI;
- add Docker and reproducibility documentation;
- document limitations and hardware-specific caveats;
- replace resume placeholders only with measured results.

Gate:

- merge requests run lint and tests;
- a deliberately slower benchmark fails the regression job;
- CI publishes an auditable report;
- README commands reproduce the documented workflow, including manual ARM
  collection and CI validation of committed ARM JSON.

## Required artifacts

```text
artifacts/
├── models/
├── metrics/
├── plots/
└── reports/
```

Each metric file must include the configuration path, model checksum, runtime,
device, thread count, timestamp, and test split identifier.

## Suggested milestone sequence

1. Data loader and MFCC tests.
2. DS-CNN forward pass and training smoke test.
3. Baseline accuracy report.
4. ONNX export and numerical validation.
5. TFLite export and INT8 calibration.
6. Benchmark harness with JSON schema.
7. x86 benchmark matrix.
8. ARM benchmark run or documented hardware execution.
9. GitLab CI regression gate.
10. Final report and README.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Preprocessing drift | Serialize shared feature settings and test cross-runtime inputs |
| Unsupported export operator | Keep the model simple and test export early |
| Noisy latency measurements | Pin threads, warm up, repeat, and report distributions |
| ARM result not reproducible | Record board/CPU/OS/runtime details and provide a script |
| Quantization accuracy loss | Use representative calibration data and a fixed test split |
| CI too slow | Use fixture data and smoke iterations; reserve full benchmarks for scheduled/manual jobs |
| Untrustworthy resume metrics | Generate metrics directly from versioned result artifacts |

## Final success criteria

- deterministic dataset and MFCC preprocessing;
- compact DS-CNN with reproducible training;
- PyTorch, ONNX Runtime, and TFLite parity checks;
- FP32 and INT8 artifacts;
- x86 and ARM benchmark support;
- p50/p95 latency, memory, size, and accuracy reports;
- GitLab CI regression protection;
- complete reproducibility and limitations documentation.
