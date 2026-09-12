"""Measured state freshness and residual-span observations; no assumed gains."""
import json
from pathlib import Path
import torch
from .config import checkpoint_path
from .data import read_manifest
from .io import load_backbone, attach, prepare_inputs
from .checkpoints import load_checkpoint, file_sha256


def residual_structure(ctrl):
    rows = []
    eps = ctrl.cfg.epsilon
    for layer, decision in ctrl.ctx.routing_log.items():
        if layer >= ctrl.decisions[-1]:
            continue
        target = (ctrl.teacher_states[layer+1]-ctrl.teacher_states[layer]).float()[0]
        active = decision.selected_mask[0].cpu()
        if not active.any() or active.all():
            continue
        a, s = target[active], target[~active]
        an = torch.nn.functional.normalize(a, dim=-1, eps=eps)
        sn = torch.nn.functional.normalize(s, dim=-1, eps=eps)
        _, singular, vh = torch.linalg.svd(a, full_matrices=False)
        rank = min(ctrl.cfg.bottleneck_dim, vh.shape[0])
        basis = vh[:rank]
        approx = (s@basis.T)@basis
        error = ((approx-s).square().mean(-1)/(s.square().mean(-1)+eps)).mean().item()
        energy = (singular[:rank].square().sum()/(singular.square().sum()+eps)).item()
        rows.append(dict(layer=layer, rank=rank, active_count=int(active.sum()), skipped_count=int((~active).sum()),
                         maximum_residual_cosine_mean=(sn@an.T).max(-1).values.mean().item(),
                         active_rank_energy=energy, skipped_span_relative_mse=error))
    return rows


def diagnose(cfg, manifest, out, limit=32):
    rows = read_manifest(manifest)[:limit]
    model, processor = load_backbone(cfg)
    ctrl = attach(cfg, model, training=True)
    path = checkpoint_path(cfg)
    load_checkpoint(path, ctrl.bank, ctrl.signature)
    ctrl.bank.eval()
    ctrl.capture_diagnostics = True
    result = dict(checkpoint_sha256=file_sha256(path), samples=[],
                  interpretation="Observations only. They do not establish accuracy improvement or causal output sensitivity.")
    for row in rows:
        batch = prepare_inputs(model, processor, row)
        record = dict(sample_id=row["sample_id"])
        with ctrl.using("dense"), torch.no_grad():
            model(**batch, use_cache=False, logits_to_keep=1)
        with ctrl.using("identity"), torch.no_grad():
            original = model(**batch, use_cache=False, logits_to_keep=1)
            record["original_freshness"] = list(ctrl.diagnostics)
            record["residual_structure"] = residual_structure(ctrl)
            reference = original.logits[:,-1].float()
        with ctrl.using("inference"), torch.no_grad():
            ghost = model(**batch, use_cache=False, logits_to_keep=1)
            record["ghost_freshness"] = list(ctrl.diagnostics)
            record["ghost_counts"] = dict(ctrl.current_stats)
            record["last_logit_relative_change"] = ((ghost.logits[:,-1].float()-reference).square().mean()/(reference.square().mean()+1e-6)).item()
        result["samples"].append(record)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+"\n")
    return dict(samples=len(rows), output=str(out))
