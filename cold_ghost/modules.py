"""Block-shared Self Evolution + Active Residual Prototype Transport."""
from __future__ import annotations
from contextlib import nullcontext
import math
import torch
from torch import nn
from .config import GhostConfig


def precision(x):
    if x.dtype in (torch.float16, torch.bfloat16) and x.device.type in ("cpu", "cuda"):
        return torch.autocast(x.device.type, dtype=x.dtype)
    return nullcontext()


class GhostBlock(nn.Module):
    def __init__(self, hidden_dim: int, cfg: GhostConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.bottleneck_dim
        self.down = nn.Linear(hidden_dim, d, bias=False)
        self.up = nn.Linear(d, hidden_dim, bias=False)
        self.residual = nn.Linear(hidden_dim, d, bias=False)
        self.prototype_key = nn.Linear(hidden_dim, d, bias=False)
        self.queries = nn.Parameter(torch.empty(cfg.num_prototypes, d))
        self.ghost_query = nn.Linear(hidden_dim, d, bias=False)
        self.ghost_key = nn.Linear(d, d, bias=False)
        self.ghost_value = nn.Linear(d, d, bias=False)
        self.context_up = nn.Linear(d, hidden_dim, bias=False)
        self.gate = nn.Linear(2*d + 1, 1)
        # z + normalized score + normalized margin + age + next-decision distance
        self.reactivate = nn.Linear(d + 4, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.queries, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.context_up.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        nn.init.zeros_(self.reactivate.weight)
        nn.init.zeros_(self.reactivate.bias)

    def age_feature(self, age, like):
        return age.clamp(max=self.cfg.max_skip_age).to(like.dtype).unsqueeze(-1) / self.cfg.max_skip_age

    def reactivation_logits(self, norm_skip, scores, threshold, score_mass, age, distance):
        with precision(norm_skip):
            z = self.down(norm_skip)
            scalars = torch.stack((scores / score_mass, (scores-threshold) / score_mass,
                                   age.clamp(max=self.cfg.max_skip_age).float() / self.cfg.max_skip_age,
                                   torch.full_like(scores, float(distance))), dim=-1).to(z.dtype)
            logits = self.reactivate(torch.cat((z, scalars), dim=-1)).squeeze(-1)
        return logits.float(), z

    def predict(self, norm_ghost, norm_full, full_residual, age, z=None):
        """Norm inputs use the CURRENT frozen decoder input_layernorm."""
        if norm_full.shape[-2] == 0:
            raise ValueError("Residual prototypes require at least one Full visual token")
        with precision(norm_ghost):
            if z is None:
                z = self.down(norm_ghost)
            self_delta = self.up(torch.nn.functional.silu(z))
            if self.cfg.ablation == "self_only":
                # A true local-only ablation: neither residual retrieval NOR gate
                # receives active-context information.
                context = torch.zeros_like(z)
                context_delta = torch.zeros_like(self_delta)
            else:
                key = self.prototype_key(norm_full)
                code = self.residual(full_residual)
                queries = self.queries.to(key.dtype)
                beta = torch.softmax((queries @ key.transpose(-1, -2)).float() / math.sqrt(key.shape[-1]), dim=-1).to(code.dtype)
                prototypes = beta @ code
                q = self.ghost_query(norm_ghost)
                k, value = self.ghost_key(prototypes), self.ghost_value(prototypes)
                alpha = torch.softmax((q @ k.transpose(-1, -2)).float() / math.sqrt(q.shape[-1]), dim=-1).to(value.dtype)
                context = alpha @ value
                context_delta = self.context_up(context)
            gate = torch.sigmoid(self.gate(torch.cat((z, context, self.age_feature(age, z)), dim=-1)))
            if self.cfg.ablation == "context_only":
                self_delta = self_delta * 0
            delta = gate * (self_delta + context_delta)
        return delta.to(norm_ghost.dtype)


class GhostBank(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, decisions: list[int], cfg: GhostConfig):
        super().__init__()
        if len(decisions) < 2 or decisions != sorted(set(decisions)):
            raise ValueError("Recoverable Ghost routing requires >=2 strictly increasing decision layers")
        if min(decisions) < 0 or max(decisions) >= num_layers:
            raise ValueError("Decision layers are outside the decoder")
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.decisions = tuple(decisions)
        indices = sorted({i // cfg.block_share_span for i in range(decisions[0], decisions[-1])})
        self.blocks = nn.ModuleDict({str(i): GhostBlock(hidden_dim, cfg) for i in indices})

    def for_layer(self, layer_idx):
        if not self.decisions[0] <= layer_idx < self.decisions[-1]:
            raise ValueError("No Ghost update before the first or after the final decision")
        return self.blocks[str(layer_idx // self.cfg.block_share_span)]

    def differentiable_zero(self):
        return next(self.parameters()).sum() * 0

    def optimizer_groups(self):
        decay, other = [], []
        for name, parameter in self.named_parameters():
            (decay if name.endswith("weight") and parameter.ndim == 2 else other).append(parameter)
        return [{"params": decay, "weight_decay": 0.01}, {"params": other, "weight_decay": 0.0}]
