import json
import wave
from pathlib import Path

import numpy as np
import torch
import yaml

from kws_benchmark.training.evaluate import evaluate
from kws_benchmark.training.model import DSCNN
from kws_benchmark.training.train import train


def test_model_roundtrip(tmp_path):
    torch.set_num_threads(1)
    model = DSCNN().eval()
    assert 10000 < sum(p.numel() for p in model.parameters()) < 100000
    inputs = torch.randn(2, 1, 101, 40)
    expected = model(inputs)
    assert expected.shape == (2, 12)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    restored = DSCNN().eval()
    restored.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    torch.testing.assert_close(expected, restored(inputs), rtol=0, atol=0)


def test_training_evaluation_reproducible(tmp_path):
    splits = {}
    for split in ("train", "validation", "test"):
        splits[split] = []
        for i, label in enumerate(("up", "down")):
            filename = f"{split}-{i}.wav"
            audio = (12000 * np.sin(np.arange(16000) * (0.1 + i * 0.1))).astype("<i2")
            with wave.open(str(tmp_path / filename), "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(audio.tobytes())
            splits[split].append({"path": filename, "label": label})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": "synthetic-fixture", "splits": splits}))
    settings = yaml.safe_load(Path("configs/train.yaml").read_text())
    settings.update(
        dataset_dir=str(tmp_path),
        manifest=str(manifest),
        output_dir=str(tmp_path / "output"),
        epochs=2,
        batch_size=2,
        candidates=[{"channels": 4, "blocks": 1}],
        accuracy_target=0.0,
    )
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump(settings))
    first = train(config)
    weights = torch.load(tmp_path / "output/baseline.pt", weights_only=True)["state_dict"]
    report = evaluate(tmp_path / "output/baseline.pt")
    assert sum(map(sum, report["confusion_matrix"])) == 2
    assert report["accuracy_target_met"]
    second = train(config)
    assert first["candidates"] == second["candidates"]
    repeated = torch.load(tmp_path / "output/baseline.pt", weights_only=True)["state_dict"]
    for key in weights:
        torch.testing.assert_close(weights[key], repeated[key], rtol=0, atol=0)
