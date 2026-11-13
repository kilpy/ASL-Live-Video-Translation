"""EDA over the How2Sign annotation CSVs. Run with `uv run python -m asl_translate.eda`.

Works on whichever splits are downloaded; cross-split checks need at least two.
"""
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from asl_translate.config import REPO_ROOT, load_config
from asl_translate.data import annotation_path, load_annotations, tokenize  # noqa: F401

REPORT_DIR = REPO_ROOT / "reports" / "eda"
VOCAB_SIZES = [500, 1000, 2000, 5000, 10000]  # 1000 is the Experiment Card's target
INK, MUTED, MARK = "#1f1f1f", "#6b6b6b", "#2a78b5"


def load_splits(data: dict) -> dict[str, pd.DataFrame]:
    """Load annotation splits that are present locally."""
    present = [s for s in data["splits"] if annotation_path(data, s).exists()]
    return {s: load_annotations(data, s) for s in present}


def percentiles(values: pd.Series) -> dict[str, float]:
    """Summarize a series with selected percentiles and its maximum."""
    q = values.quantile([0.05, 0.5, 0.95, 0.99])
    return {f"p{int(k * 100)}": round(float(v), 2) for k, v in q.items()} | {
        "max": round(float(values.max()), 2)
    }


def vocab_coverage(df: pd.DataFrame, ranked: list[str], size: int) -> dict[str, float]:
    """Share of tokens, and of whole sentences, expressible with the top `size` words."""
    vocab = set(ranked[:size])
    tokens = sum(len(t) for t in df["TOKENS"])
    return {
        "tokens": round(sum(w in vocab for t in df["TOKENS"] for w in t) / tokens, 4),
        "sentences": round(float(df["TOKENS"].map(lambda t: set(t) <= vocab).mean()), 4),
    }


def summarize(frames: dict[str, pd.DataFrame], fps: int) -> dict:
    # Vocabulary comes from train alone so eval splits can't leak into it.
    """Calculate per-split statistics, vocabulary coverage, and cross-split overlap."""
    vocab_split = "train" if "train" in frames else next(iter(frames))
    counts = Counter(w for t in frames[vocab_split]["TOKENS"] for w in t)
    ranked = [w for w, _ in counts.most_common()]

    summary = {
        "splits_present": list(frames),
        "columns": [c for c in next(iter(frames.values())).columns if c not in ("DURATION", "TOKENS")],
        "vocab_source_split": vocab_split,
        "vocab_size": len(ranked),
        "top_words": ranked[:20],
        "per_split": {},
        "cross_split": {},
    }
    total = sum(len(df) for df in frames.values())
    for split, df in frames.items():
        summary["per_split"][split] = {
            "sentences": len(df),
            "share_of_present_sentences": round(len(df) / total, 3),
            "videos": int(df["VIDEO_ID"].nunique()),
            "sentences_per_video": percentiles(df.groupby("VIDEO_ID").size()),
            "empty_sentences": int((df["SENTENCE"].str.strip() == "").sum()),
            "duplicate_sentence_ids": int(df["SENTENCE_NAME"].duplicated().sum()),
            "repeated_sentence_text": int(df["SENTENCE"].duplicated().sum()),
            "nonpositive_durations": int((df["DURATION"] <= 0).sum()),
            "total_hours": round(float(df["DURATION"].sum()) / 3600, 2),
            "duration_seconds": percentiles(df["DURATION"]),
            f"frames_at_{fps}fps": percentiles(df["DURATION"] * fps),
            "words_per_sentence": percentiles(df["TOKENS"].map(len)),
            "vocab_coverage": {
                size: vocab_coverage(df, ranked, size) for size in VOCAB_SIZES
            },
        }
    for a, b in combinations(frames, 2):
        summary["cross_split"][f"{a}/{b}"] = {
            "shared_videos": len(set(frames[a]["VIDEO_ID"]) & set(frames[b]["VIDEO_ID"])),
            "shared_sentence_text": len(set(frames[a]["SENTENCE"]) & set(frames[b]["SENTENCE"])),
        }
    return summary


def plot(df: pd.DataFrame, ranked: list[str], split: str, out: Path) -> None:
    """Save duration, sentence length, and vocabulary coverage plots for a split."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [
        (axes[0], df["DURATION"].clip(upper=df["DURATION"].quantile(0.99)), "Clip duration (s)"),
        (axes[1], df["TOKENS"].map(len).clip(upper=df["TOKENS"].map(len).quantile(0.99)),
         "Words per sentence"),
    ]
    for ax, values, label in panels:
        ax.hist(values, bins=30, color=MARK, rwidth=0.9)
        ax.axvline(values.median(), color=INK, linewidth=1, linestyle="--")
        ax.annotate(f"median {values.median():.1f}", (values.median(), ax.get_ylim()[1]),
                    xytext=(5, -12), textcoords="offset points", color=INK, fontsize=9)
        ax.set_xlabel(f"{label} (capped at p99)", color=MUTED)
        ax.set_ylabel("Sentences", color=MUTED)

    ax = axes[2]
    sizes = [s for s in VOCAB_SIZES if s <= len(ranked)] + [len(ranked)]
    shares = [vocab_coverage(df, ranked, s)["sentences"] * 100 for s in sizes]
    ax.plot(sizes, shares, color=MARK, linewidth=2, marker="o", markersize=8)
    ax.set_xscale("log")
    ticks = [s for s in VOCAB_SIZES if s <= len(ranked)]  # the full-vocabulary point stays unlabeled
    ax.set_xticks(ticks, [f"{s:,}" for s in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 105)
    ax.set_xlabel("Vocabulary size (most frequent words)", color=MUTED)
    ax.set_ylabel("Sentences fully covered (%)", color=MUTED)
    if 1000 in sizes:
        share = shares[sizes.index(1000)]
        ax.annotate(f"{share:.0f}% at 1000 words", (1000, share), xytext=(8, -14),
                    textcoords="offset points", color=INK, fontsize=9)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")
        ax.tick_params(colors=MUTED)
        ax.grid(axis="y", color="#eeeeee")
        ax.set_axisbelow(True)
    fig.suptitle(f"How2Sign annotations: {split} split", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    """Generate annotation statistics and plots for available splits."""
    data = load_config()["data"]
    frames = load_splits(data)
    if not frames:
        raise SystemExit("No annotations found; run `python -m asl_translate.data annotations`")
    summary = summarize(frames, data["fps"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    split = summary["vocab_source_split"]
    counts = Counter(w for t in frames[split]["TOKENS"] for w in t)
    plot(frames[split], [w for w, _ in counts.most_common()], split, REPORT_DIR / "overview.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
