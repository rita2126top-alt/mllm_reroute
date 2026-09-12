"""Reuse the original lmms-eval evaluator and family-specific RefCOCO presets."""
from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
from .config import ROOT, checkpoint_path, apply_overrides
from .io import attach


@contextmanager
def temporary_attribute(module, name, value):
    previous = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, previous)


def metadata(ctrl):
    if ctrl is None:
        return {"enabled": False}
    return dict(enabled=True, signature=ctrl.signature,
                checkpoint_sha256=getattr(ctrl, "checkpoint_sha256", None),
                training=getattr(ctrl, "checkpoint_metadata", {}).get("training"),
                trainable_adapter_parameters=sum(p.numel() for p in ctrl.bank.parameters()),
                adapter_parameter_bytes=sum(p.numel()*p.element_size() for p in ctrl.bank.parameters()),
                original_kv_writes=False, last_prefill=dict(ctrl.current_stats),
                scope="Counters describe the last prefill, not the entire benchmark")


def compose_eval(name, task, limit=None, checkpoint=None, ablation=None):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(ROOT/"configs"), version_base="1.3"):
        cfg = compose(config_name=name.removesuffix(".yaml"), overrides=[f"eval={task}"])
    if limit is not None:
        if limit <= 0:
            raise ValueError("eval.limit must be positive")
        OmegaConf.update(cfg, "eval.limit", limit, force_add=True)
    apply_overrides(cfg, checkpoint, ablation)
    if int(cfg.eval.batch_size) != 1:
        raise ValueError("Use the original compact batch_size=1")
    return cfg


def evaluate(name, task, out, limit=None, checkpoint=None, ablation=None):
    from scripts import run_eval as upstream
    from omegaconf import OmegaConf
    cfg = compose_eval(name, task, limit, checkpoint, ablation)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"results.json").exists():
        raise FileExistsError(f"Results already exist in {out}; choose a new --out directory")
    ghost_enabled = cfg.get("ghost", {}).get("enabled", False)
    if ghost_enabled and not checkpoint_path(cfg).is_file():
        raise FileNotFoundError(f"Train the Ghost adapter before evaluation: {checkpoint_path(cfg)}")
    holder = {}
    original_patch = upstream.patch_model_for_routing
    def patch(model, router, dispatcher, action, model_family):
        if not ghost_enabled:
            return original_patch(model, router, dispatcher, action, model_family)
        # attach builds the same original router from the same config.
        ctrl = attach(cfg, model, training=False)
        holder["ctrl"] = ctrl
        return ctrl.ctx
    (out/"resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    try:
        with temporary_attribute(upstream, "patch_model_for_routing", patch):
            results = upstream.run_lmms_eval(cfg)
        # log_results already implements the original metric flattening and global JSONL.
        old = Path.cwd()
        try:
            os.chdir(out)
            upstream.log_results(cfg, results)
        finally:
            os.chdir(old)
        (out/"raw_results.json").write_text(json.dumps(results, indent=2, default=str))
        info = dict(status="completed", config=name, task=task, limit=limit, ghost=metadata(holder.get("ctrl")))
        (out/"run_manifest.json").write_text(json.dumps(info, indent=2)+"\n")
        return info
    except Exception as exc:
        (out/"failure.json").write_text(json.dumps(dict(status="failed", error=repr(exc), config=name, task=task), indent=2)+"\n")
        raise
