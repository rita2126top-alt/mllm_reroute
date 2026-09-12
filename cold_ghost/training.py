"""Online dense teacher, identity warm-up, and closed-loop Ghost rollout."""
from __future__ import annotations
import json
import math
import random
from pathlib import Path
import torch
from .config import checkpoint_path
from .checkpoints import save_checkpoint, load_checkpoint
from .data import audit_data
from .io import load_backbone, attach, prepare_inputs
from .losses import compute_losses, TrainingTrace


class SampleOrder:
    def __init__(self, size, seed=42):
        if size < 1:
            raise ValueError("Empty training data")
        self.size, self.seed, self.epoch, self.order = size, seed, -1, None
    def at(self, offset):
        epoch, index = divmod(offset, self.size)
        if self.epoch != epoch:
            g = torch.Generator().manual_seed(self.seed+epoch)
            self.order = torch.randperm(self.size, generator=g).tolist()
            self.epoch = epoch
        return self.order[index]


def sample_loss(model, ctrl, batch, phase):
    with ctrl.using("dense"), torch.no_grad():
        output = model(**batch, use_cache=False, logits_to_keep=1)
        del output
    with ctrl.using(phase):
        output = model(**batch, use_cache=False, logits_to_keep=1)
        del output
    return compute_losses(ctrl.trace, ctrl.bank, phase)


def clear_sample(ctrl):
    ctrl.trace = TrainingTrace()
    ctrl.teacher_states = {}
    ctrl.deferred_hidden = None
    ctrl.ctx._stage_deferred_hidden = None


def validate(model, ctrl, records, prepare):
    was_training = ctrl.bank.training
    ctrl.bank.eval()
    totals = {k: 0.0 for k in ("delta", "direction", "reactivation", "freshness", "total")}
    with torch.no_grad():
        for row in records:
            losses = sample_loss(model, ctrl, prepare(row), "rollout")
            for key, value in losses.items():
                totals[key] += float(value.detach())
            clear_sample(ctrl)
    ctrl.bank.train(was_training)
    return {k: v/len(records) for k, v in totals.items()}


def accumulated_update(ctrl, rows, loss_fn, optimizer, scaler, *, context="", max_retries=16):
    """AMP overflow recovery without silently dropping a training accumulation window."""
    parameters = list(ctrl.bank.parameters())
    for attempt in range(max_retries+1):
        optimizer.zero_grad(set_to_none=True)
        metrics = {k: 0.0 for k in ("delta", "direction", "reactivation", "freshness", "total")}
        for micro, row in enumerate(rows):
            try:
                losses = loss_fn(row)
                if not bool(torch.isfinite(losses["total"])):
                    raise FloatingPointError("Non-finite forward loss; reducing AMP scale cannot fix it")
                scaler.scale(losses["total"]/len(rows)).backward()
            except Exception as exc:
                raise RuntimeError(f"{context} micro={micro} sample_id={row.get('sample_id')} failed; no sample was silently skipped") from exc
            for key, value in losses.items():
                metrics[key] += float(value.detach())/len(rows)
            clear_sample(ctrl)
        scaler.unscale_(optimizer)
        gradients = [p.grad for p in parameters if p.grad is not None]
        finite = gradients and bool(torch.stack([g.isfinite().all() for g in gradients]).all())
        if not finite:
            if not scaler.is_enabled() or attempt == max_retries:
                raise FloatingPointError(f"{context}: non-finite/missing gradients after {attempt} AMP retries; no optimizer update performed")
            scale = scaler.get_scale()
            scaler.update(new_scale=scale/2)
            print(json.dumps(dict(event="amp_overflow_retry", context=context, attempt=attempt+1,
                                  previous_scale=scale, new_scale=scaler.get_scale())), flush=True)
            continue
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        return metrics, norm, attempt
    raise AssertionError("Unreachable AMP retry state")


def train_loop(model, ctrl, train, val, prepare, out_dir, data_report, *, resume=None,
               warmup_updates=1000, rollout_updates=2000, accumulation=8,
               validation_every=100, debug=False):
    """Testable trainer. CLI fixes the research defaults; overrides are test-only."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    ctrl.bank.train()
    if any(p.requires_grad for n,p in model.named_parameters() if not n.startswith("_cold_ghost_bank")):
        raise RuntimeError("Backbone must be frozen")
    payload = None
    if resume:
        payload = load_checkpoint(resume, ctrl.bank, ctrl.signature, for_evaluation=False)
        if payload["extra"].get("data_report") != data_report:
            raise ValueError("Resume data identity differs from the checkpoint; refusing silent data changes")
        if payload["training"]["debug"] != debug:
            raise ValueError("Do not resume a debug run as a research run")
        torch.set_rng_state(payload["rng_cpu"])
        if torch.cuda.is_available() and payload.get("rng_cuda"):
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
    elif (out_dir/"last.pt").exists() or (out_dir/"best.pt").exists():
        raise FileExistsError(f"Training output exists: {out_dir}. Use --resume {out_dir/'last.pt'} or a new output directory.")
    best = float(payload["extra"].get("best_validation", math.inf)) if payload else math.inf
    warmup_completed = payload["training"].get("warmup_completed", 0) if payload else 0
    seed = 42
    order = SampleOrder(len(train), seed)
    last_metrics = {}
    for phase, updates, lr in (("warmup", warmup_updates, 1e-4), ("rollout", rollout_updates, 3e-5)):
        if payload and payload["training"]["phase"] == "rollout" and phase == "warmup":
            continue
        optimizer = torch.optim.AdamW(ctrl.bank.optimizer_groups(), lr=lr, betas=(.9,.999), eps=1e-8)
        start = 0
        if payload and payload["training"]["phase"] == phase:
            start = int(payload["training"]["step"])
            if "optimizer" not in payload:
                raise ValueError("Resume requires last.pt/warmup.pt with optimizer state, not an adapter-only checkpoint")
            optimizer.load_state_dict(payload["optimizer"])
        enabled = next(model.parameters()).device.type == "cuda" and next(model.parameters()).dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        if payload and payload["training"]["phase"] == phase and payload["extra"].get("scaler"):
            scaler.load_state_dict(payload["extra"]["scaler"])
        if start > updates:
            raise ValueError("Resume step exceeds this phase's requested updates")
        for step in range(start, updates):
            # Retry the SAME accumulation window if FP16 loss scaling overflows.
            # A failed AMP step is never counted as a completed optimizer update.
            rows = [train[order.at(step*accumulation+micro)] for micro in range(accumulation)]
            metrics, norm, overflow_retries = accumulated_update(
                ctrl, rows, lambda row: sample_loss(model, ctrl, prepare(row), phase), optimizer, scaler,
                context=f"{phase} step={step+1}")
            completed = step+1
            record = dict(phase=phase, update=completed, learning_rate=lr, grad_norm=float(norm), overflow_retries=overflow_retries, **metrics)
            if phase == "rollout" and (completed % validation_every == 0 or completed == updates):
                validation = validate(model, ctrl, val, prepare)
                record["validation"] = validation
                if validation["total"] < best:  # equality keeps the earlier checkpoint
                    best = validation["total"]
                    save_checkpoint(out_dir/"best.pt", ctrl.bank, ctrl.signature, phase=phase, step=completed,
                                    validated=True, debug=debug, warmup_completed=warmup_completed,
                                    extra=dict(data_report=data_report, validation=validation, best_validation=best))
            with (out_dir/"training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record)+"\n")
            print(json.dumps(record), flush=True)
            if phase == "warmup":
                warmup_completed = completed
            if completed % validation_every == 0 or completed == updates:
                extra = dict(data_report=data_report, best_validation=best, scaler=scaler.state_dict(),
                             accumulation=accumulation, completed_sample_offset=completed*accumulation)
                save_checkpoint(out_dir/"last.pt", ctrl.bank, ctrl.signature, phase=phase, step=completed,
                                debug=debug, warmup_completed=warmup_completed, optimizer=optimizer, extra=extra)
                if phase == "warmup" and completed == updates:
                    save_checkpoint(out_dir/"warmup.pt", ctrl.bank, ctrl.signature, phase=phase, step=completed,
                                    debug=debug, warmup_completed=warmup_completed, optimizer=optimizer, extra=extra)
            last_metrics = record
        # Explicit optimizer reset at the phase transition is intentional.
        payload = None
        del optimizer
    summary = dict(completed=True, debug=debug, warmup_updates=warmup_updates, rollout_updates=rollout_updates,
                   best_validation=best, selected_checkpoint=str(out_dir/"best.pt"),
                   data_report=data_report, last_metrics=last_metrics)
    (out_dir/"training_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def train(cfg, train_path, val_path, eval_index, out=None, resume=None, debug_updates=None):
    if not cfg.get("ghost", {}).get("enabled", False):
        raise ValueError("Train requires an enabled Ghost experiment config")
    # Before model downloads/GPU allocation, certify data isolation.
    train_rows, val_rows, report = audit_data(train_path, val_path, eval_index)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    model, processor = load_backbone(cfg)
    ctrl = attach(cfg, model, training=True)
    out = Path(out) if out else checkpoint_path(cfg).parent
    if debug_updates is not None and debug_updates < 1:
        raise ValueError("debug-updates must be positive")
    kwargs = {}
    if debug_updates is not None:
        kwargs = dict(warmup_updates=debug_updates, rollout_updates=debug_updates,
                      validation_every=1, debug=True)
    return train_loop(model, ctrl, train_rows, val_rows, lambda row: prepare_inputs(model, processor, row),
                      out, report, resume=resume, **kwargs)
