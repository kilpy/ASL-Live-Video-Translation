import numpy as np
import pytest

pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from asl_translate.data import convert_partition


@pytest.mark.parametrize("source,count,detected_count", [("mediapipe", 49, 3), ("rtmw", 65, 65)])
def test_parquet_preserves_arrays_and_source_specific_shapes(tmp_path, source, count, detected_count):
    directory = tmp_path / "npz"
    directory.mkdir()
    points = np.arange(2 * count * 2, dtype=np.float16).reshape(2, count, 2) / 17
    detected = np.ones((2, detected_count), dtype=bool)
    values = {"points": points, "detected": detected}
    if source != "mediapipe":
        values["confidence"] = np.full((2, count), 0.5, dtype=np.float16)
    np.savez_compressed(directory / "example.npz", **values)
    output = tmp_path / "data.parquet"

    assert convert_partition(source, directory, output) == 1
    row = pq.read_table(output).to_pylist()[0]
    assert row["sentence_id"] == "example"
    assert set(row) == {"sentence_id", *values}
    for key, expected in values.items():
        np.testing.assert_array_equal(np.asarray(row[key], dtype=expected.dtype), expected)
