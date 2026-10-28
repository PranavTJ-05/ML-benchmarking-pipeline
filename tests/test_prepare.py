from pathlib import Path

import yaml

from kws_benchmark.data.prepare import build_manifest


def test_manifest_respects_official_splits_and_downsamples_unknown(tmp_path: Path):
    (tmp_path / "yes").mkdir()
    (tmp_path / "cat").mkdir()
    for name in ("a.wav", "b.wav"):
        (tmp_path / "yes" / name).touch()
    for name in ("c.wav", "d.wav", "e.wav", "f.wav"):
        (tmp_path / "cat" / name).touch()
    (tmp_path / "validation_list.txt").write_text("yes/b.wav\ncat/c.wav\n")
    (tmp_path / "testing_list.txt").write_text("yes/a.wav\n")
    config = yaml.safe_load(Path("configs/base.yaml").read_text())
    config["dataset"]["unknown_downsample_ratio"] = 0.5
    manifest = build_manifest(tmp_path, config, seed=1)
    assert {x["path"] for x in manifest["splits"]["test"]} == {"yes/a.wav"}
    assert {x["path"] for x in manifest["splits"]["validation"]} == {"yes/b.wav", "cat/c.wav"}
    assert sum(x["label"] == "unknown" for x in manifest["splits"]["train"]) == 2
