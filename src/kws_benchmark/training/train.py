"""Train candidates and freeze the best validation-selected checkpoint."""

import argparse
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kws_benchmark.data.features import mfcc, normalize, read_wav
from kws_benchmark.training.model import DSCNN


def load_split(root, entries, config):
    if not entries:
        raise ValueError("Every training/evaluation split must contain clips")
    features, labels = [], []
    for entry in entries:
        audio = read_wav(
            Path(root) / entry["path"],
            config["audio"]["sample_rate_hz"],
            config["audio"]["clip_duration_seconds"],
        )
        settings = config["features"]
        features.append(
            mfcc(
                audio,
                sample_rate=config["audio"]["sample_rate_hz"],
                **{k: settings[k] for k in ("n_mfcc", "n_fft", "window_ms", "hop_ms")},
            )
        )
        labels.append(config["dataset"]["classes"].index(entry["label"]))
    return np.stack(features), torch.tensor(labels, dtype=torch.long)


def scores(model, inputs, labels, classes, batch_size=64):
    model.eval()
    confusion = torch.zeros((classes, classes), dtype=torch.long)
    with torch.no_grad():
        for x, y in DataLoader(TensorDataset(inputs, labels), batch_size=batch_size):
            predicted = model(x).argmax(1)
            confusion += torch.bincount(
                y * classes + predicted, minlength=classes * classes
            ).reshape(classes, classes)
    per_class = []
    for index in range(classes):
        tp = confusion[index, index].item()
        support = confusion[index].sum().item()
        precision = tp / max(1, confusion[:, index].sum().item())
        recall = tp / max(1, support)
        per_class.append(
            {
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / max(1e-12, precision + recall),
            }
        )
    return {
        "accuracy": confusion.diag().sum().item() / len(labels),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def train(config_path):
    settings = yaml.safe_load(Path(config_path).read_text())
    config = yaml.safe_load(Path(settings["base_config"]).read_text())
    manifest_bytes = Path(settings["manifest"]).read_bytes()
    manifest = json.loads(manifest_bytes)
    splits = manifest["splits"]
    # Reject leakage, including the unfinished Phase 1 shared-noise placeholder.
    paths = [{e["path"] for e in splits[name]} for name in ("train", "validation", "test")]
    if any(paths[i] & paths[j] for i in range(3) for j in range(i)):
        raise ValueError("Manifest contains overlapping split paths; prepare isolated clips first")
    train_x, train_y = load_split(settings["dataset_dir"], splits["train"], config)
    val_x, val_y = load_split(settings["dataset_dir"], splits["validation"], config)
    train_x, mean, std = normalize(train_x)
    val_x, _, _ = normalize(val_x, mean, std)
    train_x = torch.from_numpy(train_x[:, None])
    val_x = torch.from_numpy(val_x[:, None])
    config["features"]["normalization"].update(mean=mean.tolist(), std=std.tolist())
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(settings["threads"])
    torch.use_deterministic_algorithms(True)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        commit = None
    metadata = {
        "schema_version": "1.0",
        "seed": config["seed"],
        "git_commit": commit,
        "config_path": str(config_path),
        "training_config": settings,
        "resolved_config": config,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "dataset_version": manifest["version"],
        "torch_version": str(torch.__version__),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    winner, best_accuracy, results = None, -1, []
    for candidate in settings["candidates"]:
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        model = DSCNN(classes=len(config["dataset"]["classes"]), **candidate)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
        loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=settings["batch_size"],
            shuffle=True,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
        best, stale, history = -1, 0, []
        for epoch in range(settings["epochs"]):
            model.train()
            for inputs, targets in loader:
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(inputs), targets)
                loss.backward()
                optimizer.step()
            metrics = scores(model, val_x, val_y, len(config["dataset"]["classes"]))
            history.append({"epoch": epoch + 1, **metrics})
            if metrics["accuracy"] > best:
                best, stale = metrics["accuracy"], 0
                checkpoint = {
                    "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "model_config": candidate,
                    "metadata": metadata,
                    "epoch": epoch + 1,
                }
            else:
                stale += 1
            if stale >= settings["patience"]:
                break
        torch.save(checkpoint, output / f"candidate-{len(results)}.pt")
        results.append({"model": candidate, "history": history, "best_validation_accuracy": best})
        if best > best_accuracy:
            best_accuracy, winner = best, checkpoint
    if winner is None:
        raise ValueError("At least one candidate and training epoch are required")
    torch.save(winner, output / "baseline.pt")
    (output / "resolved.yaml").write_text(yaml.safe_dump(config))
    report = {
        **metadata,
        "candidates": results,
        "selected_validation_accuracy": best_accuracy,
        "baseline_sha256": hashlib.sha256((output / "baseline.pt").read_bytes()).hexdigest(),
    }
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    train(parser.parse_args().config)


if __name__ == "__main__":
    main()
