import pandas as pd
import pytest
import yaml

from asl_translate.config import CONFIG_PATH, load_config
from asl_translate.eda import summarize, tokenize


def frame(video: str, sentences: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "VIDEO_ID": video,
            "SENTENCE_NAME": [f"{video}_{i}" for i in range(len(sentences))],
            "SENTENCE": sentences,
            "DURATION": 2.0,
            "TOKENS": [tokenize(s) for s in sentences],
        }
    )


def test_every_approach_shares_the_data_block():
    names = load_config()["approaches"]
    assert set(names) == {"CLASSICAL", "BASELINE", "DEEP"}
    assert all(load_config(n)["data"] == load_config()["data"] for n in names)


def test_approach_resolves_encoder_and_decoder_blocks():
    baseline = load_config("BASELINE")
    assert baseline["encoder"]["name"] == "pose_gru"
    assert baseline["encoder"]["features"] == "landmarks"
    assert baseline["decoder"]["name"] == "transformer"


def test_plugging_in_a_new_decoder_only_needs_config(tmp_path):
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    cfg["decoders"]["t5_small"] = {"pretrained": True}
    cfg["approaches"]["TRIAL"] = {"encoder": "pose_gru", "decoder": "t5_small"}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert load_config("TRIAL", path=path)["decoder"] == {"name": "t5_small", "pretrained": True}


def test_unknown_approach_is_rejected():
    with pytest.raises(ValueError):
        load_config("NOPE")


def test_data_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ASL_DATA_ROOT", str(tmp_path))
    assert load_config()["data"]["root"] == tmp_path


def test_tokenize_lowercases_and_keeps_contractions():
    assert tokenize("We're going, OK!") == ["we're", "going", "ok"]


def test_summary_flags_cross_split_overlap_and_uses_train_vocab():
    frames = {
        "train": frame("a", ["keep your back straight", "hold it"]),
        "val": frame("a", ["hold it", "zebra"]),
    }
    summary = summarize(frames, fps=15)
    assert summary["vocab_source_split"] == "train"
    assert summary["cross_split"]["train/val"] == {"shared_videos": 1, "shared_sentence_text": 1}
    assert summary["per_split"]["val"]["vocab_coverage"][500] == {
        "tokens": 0.6667,
        "sentences": 0.5,
    }
