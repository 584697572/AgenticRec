"""Safely inspect the complete original checkpoint; never fall back to unrestricted pickle."""
import codecs
import hashlib
import json
import pickletools
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "reproduction/logs/a1_20261001"
CHECKPOINT = ROOT / "data/raw/upstream_movie/SASRec-SASRec.pth"


def main():
    import numpy as np
    import torch
    with zipfile.ZipFile(CHECKPOINT) as archive:
        payload = archive.read(next(name for name in archive.namelist() if name.endswith("/data.pkl")))
        symbols = sorted({arg for op, arg, _ in pickletools.genops(payload) if op.name == "GLOBAL"})
    expected = {'_codecs encode', 'collections OrderedDict', 'numpy dtype',
                'numpy.core.multiarray scalar', 'torch FloatStorage', 'torch device',
                'torch._utils _rebuild_tensor_v2'}
    if set(symbols) != expected:
        raise ValueError("Unexpected checkpoint global references; refusing load")
    # NumPy 2 changed the callable's __module__; explicitly allow only the old name.
    allow = [codecs.encode, (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
             np.dtype, type(np.dtype("float64")), type(np.dtype("float32"))]
    with torch.serialization.safe_globals(allow):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    allowed = {"model", "dataset", "n_items", "n_users", "embedding_size", "max_seq_len", "hidden_size"}
    result = {"checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "weights_only": True, "unrestricted_pickle_fallback": False,
              "global_references": symbols, "config": {k: v for k, v in checkpoint["config"].items() if k in allowed},
              "state_dict_shapes": {k: list(v.shape) for k, v in checkpoint["state_dict"].items()},
              "model_executed": False, "training_title_mapping": "NOT VERIFIED"}
    (LOG / "real_checkpoint_inspection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
