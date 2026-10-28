"""Evaluate the frozen baseline on its recorded official test manifest."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from kws_benchmark.data.features import normalize
from kws_benchmark.training.model import DSCNN
from kws_benchmark.training.train import load_split, scores


def evaluate(checkpoint_path):
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    config = metadata["resolved_config"]
    settings = metadata["training_config"]
    raw = Path(settings["manifest"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata["manifest_sha256"]:
        raise ValueError("Manifest changed since training")
    torch.set_num_threads(settings["threads"])
    x, y = load_split(settings["dataset_dir"], json.loads(raw)["splits"]["test"], config)
    import numpy as np

    stats = config["features"]["normalization"]
    x, _, _ = normalize(x, np.asarray(stats["mean"]), np.asarray(stats["std"]))
    model = DSCNN(classes=len(config["dataset"]["classes"]), **checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    metrics = scores(model, torch.from_numpy(x[:, None]), y, len(config["dataset"]["classes"]))
    target = settings.get("accuracy_target")
    report = {
        **metadata,
        **metrics,
        "classes": config["dataset"]["classes"],
        "test_split": "official_test",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_size_bytes": path.stat().st_size,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "accuracy_target_met": None if target is None else metrics["accuracy"] >= target,
    }
    (path.parent / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    evaluate(parser.parse_args().checkpoint)


if __name__ == "__main__":
    main()
