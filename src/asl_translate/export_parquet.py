"""Convert published sentence-level NPZ features to Hugging Face Parquet partitions."""
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def convert_partition(source: str, source_dir: Path, output: Path) -> int:
    """Convert NPZ arrays losslessly, retaining their source-specific shape and dtype."""
    files = sorted(source_dir.glob("*.npz"))
    if not files:
        raise ValueError(f"No feature files found in {source_dir}")
    fields = [pa.field("sentence_id", pa.string()),
              pa.field("points", pa.list_(pa.list_(pa.list_(pa.float16())))),
              pa.field("detected", pa.list_(pa.list_(pa.bool_())))]
    if source != "mediapipe":
        fields.append(pa.field("confidence", pa.list_(pa.list_(pa.float16()))))
    schema = pa.schema(fields)
    keys = ["points", "detected"] + ([] if source == "mediapipe" else ["confidence"])
    count = 49 if source == "mediapipe" else 65
    detected_count = 3 if source == "mediapipe" else 65
    output.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(output, schema, compression="zstd") as writer:
        for start in range(0, len(files), 32):
            rows = []
            for path in files[start:start + 32]:
                with np.load(path, allow_pickle=False) as data:
                    points, detected = data["points"], data["detected"]
                    if (points.dtype != np.float16 or points.shape[1:] != (count, 2)
                            or detected.dtype != np.bool_
                            or detected.shape != (len(points), detected_count)):
                        raise ValueError(f"Unexpected shape or dtype in {path}")
                    if source != "mediapipe" and (data["confidence"].dtype != np.float16
                            or data["confidence"].shape != (len(points), count)):
                        raise ValueError(f"Unexpected confidence shape or dtype in {path}")
                    rows.append({"sentence_id": path.stem, **{k: data[k].tolist() for k in keys}})
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mediapipe-root", type=Path, required=True)
    parser.add_argument("--other-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for source in ("mediapipe", "vitpose", "rtmw", "dwpose"):
        for split in ("train", "val", "test"):
            if source == "mediapipe" and split == "test":
                continue
            source_dir = (args.mediapipe_root / split if source == "mediapipe"
                          else args.other_root / source / split)
            output = args.output / source / f"{split}-00000.parquet"
            print(source, split, convert_partition(source, source_dir, output), flush=True)


if __name__ == "__main__":
    main()
