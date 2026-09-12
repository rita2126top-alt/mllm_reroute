"""Four FP32 losses, with next actual routing decision as reactivation label."""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

@dataclass
class TrainingTrace:
    residuals: list = field(default_factory=list)
    reactivation: dict = field(default_factory=dict)
    full_masks: dict = field(default_factory=dict)
    freshness: list = field(default_factory=list)


def relative_error(prediction, target, epsilon):
    prediction, target = prediction.float(), target.float()
    return (prediction-target).square().mean(-1) / (target.square().mean(-1)+epsilon)


def compute_losses(trace: TrainingTrace, bank, mode: str):
    eps = bank.cfg.epsilon
    zero = bank.differentiable_zero()
    residual_terms, directions, react_terms, fresh_terms = [], [], [], []
    for prediction, target in trace.residuals:
        if prediction.numel() == 0:
            continue
        residual_terms.append(relative_error(prediction, target, eps).reshape(-1))
        p, t = prediction.float(), target.float()
        pn, tn = p.norm(dim=-1), t.norm(dim=-1)
        valid = ((pn > eps) & (tn > eps)).detach()
        if valid.any():
            directions.append((1-(p*t).sum(-1)/(pn*tn+eps))[valid])
    for layer, (logits, candidates, next_layer) in trace.reactivation.items():
        if next_layer not in trace.full_masks:
            raise RuntimeError(f"Missing actual next-decision label at {next_layer} after {layer}")
        label = trace.full_masks[next_layer].index_select(-1, candidates).float().detach()
        react_terms.append(F.binary_cross_entropy_with_logits(logits.float(), label, reduction="none").reshape(-1))
    if mode == "rollout" and bank.cfg.ablation != "no_fresh":
        for prediction, target in trace.freshness:
            if prediction.numel():
                fresh_terms.append(relative_error(prediction, target, eps).reshape(-1))
    def reduce(terms):
        return torch.cat(terms).mean() if terms else zero
    losses = {"delta": reduce(residual_terms), "direction": reduce(directions),
              "reactivation": reduce(react_terms), "freshness": reduce(fresh_terms)}
    losses["total"] = losses["delta"] + .1*losses["direction"] + .1*losses["reactivation"] + losses["freshness"]
    if not torch.isfinite(losses["total"]):
        raise FloatingPointError("Non-finite Ghost loss; sample is not silently skipped")
    return losses
