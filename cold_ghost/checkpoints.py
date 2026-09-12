"""Versioned adapter-only checkpoints; no backbone tensors or pickle classes."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(1024*1024), b""):
            digest.update(data)
    return digest.hexdigest()


def save_checkpoint(path, bank, signature, *, phase, step, validated=False,
                    debug=False, warmup_completed=0, optimizer=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "signature": signature,
        "bank": {k: v.detach().cpu() for k, v in bank.state_dict().items()},
        "training": dict(phase=phase, step=int(step), validated=bool(validated),
                         debug=bool(debug), warmup_completed=int(warmup_completed)),
        "extra": extra or {},
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, bank, signature, *, for_evaluation=True):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained Ghost checkpoint missing: {path}. Run the two training phases first; random-weight evaluation is forbidden.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != 1 or payload.get("signature") != signature:
        raise ValueError("Ghost checkpoint identity mismatch: model/width/layers/routing/ghost settings must match. Only compact vs stagewise action may differ.")
    training = payload.get("training", {})
    if for_evaluation and (training.get("phase") != "rollout" or not training.get("validated")
                           or training.get("debug") or training.get("warmup_completed", 0) < 1000):
        raise ValueError("Evaluation requires a non-debug, validation-selected rollout checkpoint after the full 1000-update warm-up")
    bank.load_state_dict(payload["bank"], strict=True)
    return payload
