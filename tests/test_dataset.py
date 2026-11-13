import pandas as pd
import pytest

from asl_translate.config import load_config
from asl_translate.data import (
    build_vocab,
    covered,
    make_splits,
    pick_by_video,
    record,
    tokenize,
    verify,
)


def frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """rows of (video_id, video_name, sentence)."""
    df = pd.DataFrame(rows, columns=["VIDEO_ID", "VIDEO_NAME", "SENTENCE"])
    df["SENTENCE_NAME"] = [f"{v}_{i}" for i, v in enumerate(df["VIDEO_NAME"])]
    df["DURATION"] = 2.0
    df["TOKENS"] = df["SENTENCE"].map(tokenize)
    return df


def test_vocab_is_ranked_by_frequency_then_alphabetically():
    train = frame([("a", "a1", "b a c"), ("a", "a1", "a b")])
    assert build_vocab(train, 2) == ["a", "b"]
    assert build_vocab(train, 10) == ["a", "b", "c"]


def test_covered_drops_unknown_words_empty_text_and_bad_durations():
    df = frame([("a", "a1", "hold it"), ("a", "a1", "zebra"), ("a", "a1", ""), ("a", "a1", "it")])
    df.loc[3, "DURATION"] = 0.0
    assert list(covered(df, {"hold", "it"})["SENTENCE"]) == ["hold it"]


def test_pick_by_video_takes_whole_videos_densest_first_and_is_seeded():
    df = frame([("x", "big", "s")] * 5 + [("y", "mid", "s")] * 3 + [("z", "small", "s")])
    picked = pick_by_video(df, 6, seed=0)
    assert set(picked["VIDEO_NAME"]) == {"big", "mid"}
    assert pick_by_video(df, 6, seed=0).equals(picked)
    assert pick_by_video(df, None, seed=0).equals(df)


def test_splits_use_train_vocab_only_and_never_share_videos():
    frames = {
        "train": frame([("a", "a1", "keep your back straight"), ("b", "b1", "hold it")]),
        "val": frame([("c", "c1", "hold it"), ("c", "c1", "zebra crossing")]),
        "test": frame([("d", "d1", "keep it straight")]),
    }
    cfg = {"vocab_size": 100, "n_train_sentences": None, "n_val_sentences": None,
           "split_seed": 0, "smoke": {"train": 1, "val": 1}}
    available = {s: set(df["VIDEO_NAME"]) for s, df in frames.items()}
    vocab, chosen = make_splits(frames, cfg, available)
    assert "zebra" not in vocab  # val words never enter the vocabulary
    assert list(chosen["val"]["SENTENCE"]) == ["hold it"]
    assert len(chosen["smoke_train"]) == 1 and set(chosen["smoke_train"]["SENTENCE_NAME"]) <= set(
        chosen["train"]["SENTENCE_NAME"])

    frames["val"].loc[0, "VIDEO_ID"] = "a"  # same source video as train
    with pytest.raises(SystemExit):
        make_splits(frames, cfg, available)


def test_sentences_from_videos_missing_in_the_mirror_are_skipped():
    frames = {s: frame([(s, f"{s}1", "hold it"), (s, f"{s}2", "hold it")])
              for s in ("train", "val", "test")}
    cfg = {"vocab_size": 10, "n_train_sentences": None, "n_val_sentences": None,
           "split_seed": 0, "smoke": {"train": 1, "val": 1}}
    available = {s: {f"{s}1"} for s in frames}
    _, chosen = make_splits(frames, cfg, available)
    assert set(chosen["val"]["VIDEO_NAME"]) == {"val1"}


def test_manifest_detects_changed_data(tmp_path, monkeypatch):
    monkeypatch.setenv("ASL_DATA_ROOT", str(tmp_path))
    data_file = load_config()["data"]["processed"] / "splits" / "val.txt"
    data_file.parent.mkdir(parents=True)
    data_file.write_text("a\n")
    manifest = tmp_path / "manifest.json"
    record([data_file], path=manifest)
    verify([data_file], path=manifest)
    data_file.write_text("b\n")
    with pytest.raises(SystemExit):
        verify([data_file], path=manifest)


def test_upload_limits_files_to_selected_features(tmp_path, monkeypatch):
    import huggingface_hub

    from asl_translate import data as data_module

    processed = tmp_path / "processed"
    for path in ("splits", "landmarks/rtmw/train"):
        (processed / path).mkdir(parents=True)
    (processed / "splits/vocab.txt").touch()
    (processed / "video_index.json").touch()
    (processed / "landmarks/rtmw/metadata.json").touch()
    calls = []

    class Api:
        def upload_folder(self, **kwargs):
            calls.append(kwargs)
            return type("Info", (), {"oid": "revision"})()

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(data_module, "verify", lambda paths: None)
    manager = data_module.DataManager({"processed": processed, "processed_hub": {"repo": "org/data"}})
    manager.upload_features("rtmw", ["train"])
    assert calls[0]["allow_patterns"] == ["splits/**", "video_index.json", "landmarks/rtmw/train/**", "landmarks/rtmw/metadata.json"]

    with pytest.raises(SystemExit, match="Missing"):
        manager.upload_features("rtmw", ["test"])
    assert len(calls) == 1
