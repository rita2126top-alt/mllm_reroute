"""Validated fixed method configuration and checkpoint identity."""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class GhostConfig:
    enabled: bool = True
    budget: int = 128
    bottleneck_dim: int = 32
    num_prototypes: int = 8
    horizon_decisions: int = 1
    max_skip_age: int = 8
    block_share_span: int = 4
    partition_at_decision_only: bool = True
    update_every_layer_in_stage: bool = True
    enable_after_last_decision: bool = False
    write_original_kv: bool = False
    epsilon: float = 1e-6
    ablation: str = "full"

    def __post_init__(self):
        if self.budget < 0:
            raise ValueError("ghost.budget must be nonnegative")
        for key in ("bottleneck_dim", "num_prototypes", "max_skip_age", "block_share_span"):
            if getattr(self, key) < 1:
                raise ValueError(f"ghost.{key} must be positive")
        if self.horizon_decisions != 1 or not self.partition_at_decision_only or not self.update_every_layer_in_stage:
            raise ValueError("Only next-decision labels and decision-only partition with per-layer updates are implemented")
        if self.enable_after_last_decision or self.write_original_kv:
            raise ValueError("Ghost is disabled after the last decision and never writes original KV")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.ablation not in ("full", "self_only", "context_only", "no_fresh", "uniform_ghost"):
            raise ValueError(f"Unknown ablation: {self.ablation}")

    @classmethod
    def from_dict(cls, value):
        value = dict(value or {})
        allowed = {f.name for f in fields(cls)}
        unknown = set(value) - allowed - {"checkpoint"}
        if unknown:
            raise ValueError(f"Unknown ghost settings: {sorted(unknown)}")
        return cls(**{k: v for k, v in value.items() if k in allowed})


def read_config(name: str):
    """Resolve the original experiment/model defaults without changing any YAML."""
    from omegaconf import OmegaConf
    path = Path(name)
    if not path.is_file():
        path = ROOT / "configs" / (name.removesuffix(".yaml") + ".yaml")
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {path}")
    cfg = OmegaConf.load(path)
    model_name = None
    for entry in cfg.get("defaults", []):
        if hasattr(entry, "keys") and "/model" in entry:
            model_name = str(entry["/model"])
    if model_name is None:
        raise ValueError(f"Missing /model default in {path}")
    cfg.model = OmegaConf.load(ROOT / "configs/model" / f"{model_name}.yaml")
    return cfg


def build_router(cfg):
    # These are the ORIGINAL selectors, not approximations or replacements.
    from models.router import FastVRouter, PDropRouter
    r = cfg.routing
    if r.method == "none":
        return None
    if r.method == "fastv":
        return FastVRouter(scoring_layer=int(r.scoring_layer), keep_ratio=float(r.keep_ratio))
    if r.method == "pdrop":
        return PDropRouter(list(r.drop_layers), list(r.keep_ratios), monotonic=bool(r.get("monotonic", True)))
    raise ValueError(f"Unsupported original routing.method: {r.method}")


def family(cfg):
    kind = str(cfg.model.lmms_model_type)
    if kind == "llava_hf":
        return "llava"
    if kind == "qwen2_5_vl":
        return "qwen25vl"
    raise ValueError(f"This extension supports the original two model families, not {kind}")


def identity(cfg, hidden_dim: int, num_layers: int):
    """Compact and stagewise share weights; different schedules do NOT."""
    from omegaconf import OmegaConf
    routing = OmegaConf.to_container(cfg.routing, resolve=True)
    routing.pop("action", None)
    ghost = GhostConfig.from_dict(cfg.get("ghost", {}))
    result = dict(schema=1, model=str(cfg.model.pretrained), family=family(cfg),
                  hidden_dim=int(hidden_dim), num_layers=int(num_layers),
                  routing=routing, ghost=asdict(ghost))
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    return {**result, "sha256": hashlib.sha256(encoded).hexdigest()}


def checkpoint_path(cfg):
    value = cfg.get("ghost", {}).get("checkpoint")
    if not value:
        raise ValueError("ghost.checkpoint is required for evaluation/profiling; random Ghost evaluation is forbidden")
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def apply_overrides(cfg, checkpoint=None, ablation=None):
    """Keep ablation checkpoints separate from the fixed full-method protocol."""
    from omegaconf import OmegaConf
    if checkpoint is not None:
        OmegaConf.update(cfg, "ghost.checkpoint", str(Path(checkpoint).resolve()), force_add=True)
    if ablation is not None:
        OmegaConf.update(cfg, "ghost.ablation", ablation, force_add=True)
        if ablation != "full" and checkpoint is None:
            path = str(cfg.ghost.checkpoint).replace("checkpoints/cold_ghost/", f"checkpoints/cold_ghost_ablations/{ablation}/")
            OmegaConf.update(cfg, "ghost.checkpoint", path, force_add=True)
    return cfg
