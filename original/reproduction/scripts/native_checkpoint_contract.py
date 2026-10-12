"""Verify original loader uses all real checkpoint weights and scores the catalog subset."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "reproduction/logs/a1_20261001"
sys.path.insert(0, str(ROOT / "reproduction/.runtime/InteRecAgent-compat"))


def main():
    import numpy as np
    import pandas as pd
    import torch
    from unirec.utils.general import load_model_freely
    from llm4crs.environ_variables import MODEL_CKPT_FILE, GAME_INFO_FILE
    safe_inspection = json.loads((LOG / "real_checkpoint_inspection.json").read_text(encoding="utf-8"))
    assert safe_inspection["weights_only"] and not safe_inspection["unrestricted_pickle_fallback"]
    assert hashlib.sha256(Path(MODEL_CKPT_FILE).read_bytes()).hexdigest() == safe_inspection["checkpoint_sha256"]
    torch.set_num_threads(4)
    # Original Torch 1.13 loader is only called in run_isolated's credential-free process,
    # after complete CRC/SHA and the restricted weights-only inspection succeeded.
    model, config = load_model_freely(MODEL_CKPT_FILE, "cpu")
    checkpoint = torch.load(MODEL_CKPT_FILE, map_location="cpu")
    original_state, loaded_state = checkpoint["state_dict"], model.state_dict()
    assert set(original_state) == set(loaded_state), "Original loader silently omitted weight keys"
    assert all(torch.equal(value.cpu(), loaded_state[name].cpu()) for name, value in original_state.items())
    ids = pd.read_feather(GAME_INFO_FILE)["id"].to_numpy()
    assert len(ids) == 9888 and ids.min() == 1 and ids.max() == 9888
    scores = model.predict({"item_seq": torch.tensor([[1062, 1023]]),
                            "item_seq_len": torch.tensor([2], dtype=torch.int),
                            "item_id": torch.tensor(ids, dtype=torch.long)})
    assert scores.shape == (1, 9888) and np.isfinite(scores).all()
    result = {"success": True, "checkpoint_sha256": safe_inspection["checkpoint_sha256"],
              "original_loader": "unirec.utils.general.load_model_freely", "torch": torch.__version__,
              "state_key_count": len(original_state), "all_keys_and_weights_exactly_equal": True,
              "missing_keys": [], "unexpected_keys": [], "scores_shape": list(scores.shape),
              "all_scores_finite": True, "training_mode_preserved": model.training,
              "title_mapping": "NOT VERIFIED", "recommendation_quality": "NOT EVALUATED"}
    (LOG / "native_checkpoint_contract.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
