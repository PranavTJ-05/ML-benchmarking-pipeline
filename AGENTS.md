# AGENTS.md

## Project

**Edge Keyword Spotter - ML Benchmarking Pipeline**

This repository builds a reproducible keyword-spotting model and a benchmark
harness for PyTorch, ONNX Runtime, and TFLite across x86 and ARM targets.

## North-star outcome

Every reported number must be produced by a checked-in configuration and a
repeatable command. Do not invent resume metrics or benchmark results.

The system must measure:

- accuracy and per-class metrics;
- model size and parameter count;
- p50 and p95 single-item latency;
- peak memory where supported;
- runtime, device, OS, Python, and thread configuration;
- FP32 versus INT8 accuracy and performance trade-offs.

## Operating rules for agents

1. Read `roadmap.md` and the relevant configuration before changing code.
2. Preserve the existing public interfaces and configuration schema unless the
   task explicitly changes them.
3. Keep preprocessing identical across PyTorch, ONNX Runtime, and TFLite.
4. Use deterministic seeds and record dataset split/version information.
5. Never compare runtimes using different input features or hidden defaults.
6. Never weaken, delete, or skip tests to make a check pass.
7. Do not claim a test, benchmark, export, or CI check passed unless it ran.
8. Keep generated model files and large datasets out of Git unless explicitly
   requested; publish them as CI artifacts instead.
9. Do not store credentials, tokens, private audio, or machine-specific paths
   in repository files.
10. Keep changes focused and update documentation/configuration when behavior
    changes.

## Standard development loop

```text
Understand -> configure -> implement -> test -> benchmark -> document
```

For every change:

1. Inspect callers, tests, and configuration.
2. Make the smallest complete change.
3. Run the narrowest relevant check first.
4. Run the full affordable check suite.
5. Record any changed metric, threshold, or limitation.

## Canonical commands

Use the project environment's package runner where available.

```bash
pytest -q
pytest --cov=src --cov-report=term-missing
ruff check .
ruff format --check .
```

Expected future CLI entry points:

```bash
python -m kws_benchmark.data.prepare --config configs/base.yaml
python -m kws_benchmark.training.train --config configs/train.yaml
python -m kws_benchmark.training.evaluate --config configs/base.yaml
python -m kws_benchmark.export.export_all --config configs/export.yaml
python -m kws_benchmark.benchmark.run --config configs/benchmark_x86.yaml
python -m kws_benchmark.benchmark.report --results artifacts/metrics
```

## Definition of a valid benchmark

A benchmark is valid only when it includes:

- warmup count and measured iteration count;
- batch size and thread count;
- model artifact checksum;
- runtime and version;
- device architecture;
- raw latency samples or an auditable summary;
- accuracy evaluated on the same test split;
- timestamp and configuration path.

CI smoke benchmarks may use fewer iterations, but must clearly be labeled as
smoke measurements and must not be presented as final device results.

## Regression policy

The regression job should fail when any configured gate is exceeded:

- p95 latency increase above the configured percentage;
- accuracy drop above the configured percentage-point limit;
- model-size increase above the configured limit;
- missing or unloadable model artifact;
- PyTorch/ONNX/TFLite prediction mismatch above tolerance.

Thresholds belong in version-controlled configuration, not hard-coded in test
logic.

## Agent handoff format

When finishing a task, report:

- changed files;
- commands actually run and their results;
- benchmark configuration and environment, if relevant;
- assumptions;
- remaining risks or next steps.

## Source of truth

`roadmap.md` defines phase gates and acceptance criteria. If implementation
reveals a necessary change, update the roadmap in the same change.
