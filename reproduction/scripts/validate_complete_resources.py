"""Verify complete original movie members independently, without deserializing models."""
import hashlib
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "reproduction/logs/a1_20261001"


def main():
    inventory = {entry["name"]: entry for entry in json.loads((ROOT / "reproduction/archive_inventory.json").read_text(encoding="utf-8"))}
    records = []
    for name in ("settings.json", "columns.json", "movies.ftr", "movie_sim.npy", "SASRec-SASRec.pth"):
        path = ROOT / "data/raw/upstream_movie" / name
        entry = inventory["movie/" + name]
        digest, crc, length = hashlib.sha256(), 0, 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                length += len(chunk)
                digest.update(chunk)
                crc = zlib.crc32(chunk, crc)
        assert length == entry["bytes"] and format(crc & 0xFFFFFFFF, "08x") == entry["crc32"], name
        records.append({"name": name, "path": path.relative_to(ROOT).as_posix(), "bytes": length,
                        "sha256": digest.hexdigest(), "crc32": entry["crc32"], "status": "PASS"})
    settings = json.loads((ROOT / "data/raw/upstream_movie/settings.json").read_text(encoding="utf-8"))
    columns = json.loads((ROOT / "data/raw/upstream_movie/columns.json").read_text(encoding="utf-8"))
    assert settings["USE_COLS"] == list(columns)
    assert all((ROOT / "data/raw/upstream_movie" / settings[key]).is_file()
               for key in ("GAME_INFO_FILE", "TABLE_COL_DESC_FILE", "MODEL_CKPT_FILE", "ITEM_SIM_FILE"))
    result = {"success": True, "full_movie_members": records, "archive_downloaded": False,
              "shared_id_semantics": "NOT VERIFIED", "redistribution_license": "NOT VERIFIED",
              "source": "Fixed upstream README public volunteer-provided all_resources.zip",
              "local_download_authorized_by_user": True, "redistributed": False}
    (LOG / "complete_movie_integrity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
