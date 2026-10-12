"""A checkout includes the published cohort hashes but excludes private labels."""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agenticrec.evaluation import cohort


@pytest.mark.parametrize("tampered", [False, True])
def test_restore_private_cohorts_only_when_published_hashes_match(tmp_path, monkeypatch, tampered):
    monkeypatch.setattr(cohort, "ROOT", tmp_path)
    data = tmp_path / "data/processed/ml-1m-v1"
    data.mkdir(parents=True)
    mapping = data / "id_map.json"
    mapping.write_text(json.dumps({"user_to_model": {"11": 1},
                                   "item_to_model": {"21": 1, "22": 2}}), encoding="utf-8")
    for name, item in (("train", 21), ("eval_valid", 22), ("eval_test", 22)):
        pq.write_table(pa.Table.from_pylist([{"raw_user_id": 11, "raw_item_id": item,
                                             "rating": 5, "timestamp": 1}]), data / (name + ".parquet"))
    manifest = tmp_path / "artifacts/data_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"id_map_sha256": cohort.sha(mapping), "files": {
        name: {"sha256": cohort.sha(data / name)}
        for name in ("train.parquet", "eval_valid.parquet", "eval_test.parquet")}}), encoding="utf-8")
    cohort.freeze()
    published = tmp_path / "reproduction/native_runs/20261006_metrics/cohort_summary.json"
    private = tmp_path / "data/eval_private/ml-1m-v1"
    private.rename(tmp_path / "original_private")
    if tampered:
        value = json.loads(published.read_text(encoding="utf-8"))
        value["splits"]["test"]["sha256"] = "0" * 64
        published.write_text(json.dumps(value), encoding="utf-8")
    original_summary = published.read_bytes()
    if tampered:
        with pytest.raises(ValueError, match="published"):
            cohort.freeze()
        assert not private.exists()
    else:
        cohort.freeze()
        for name in ("valid.json", "test.json"):
            assert (private / name).read_bytes() == (tmp_path / "original_private" / name).read_bytes()
        with pytest.raises(FileExistsError):
            cohort.freeze()
    assert published.read_bytes() == original_summary
