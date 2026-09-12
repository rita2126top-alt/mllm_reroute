"""An opt-in adapter: no edits or process-global changes to models/patching.py.

The original Q/K scorer, RoutingDecision, PDropRouter and its stage cache are
called directly. Only deferred-state evolution and compact data movement are
new. All original model files remain byte-identical.
"""
from __future__ import annotations
import functools
from contextlib import contextmanager
import torch
from torch import Tensor
from models import patching as original
from models.dispatcher import TokenDispatcher
from models.router import PDropRouter
from .config import GhostConfig
from .modules import GhostBank
from .losses import TrainingTrace, relative_error


def first_hidden(result):
    return result[0] if isinstance(result, tuple) else result


def with_hidden(result, hidden):
    return (hidden, *result[1:]) if isinstance(result, tuple) else hidden


def slice_positions(position_ids, position_embeddings, kept):
    ids = None if position_ids is None else position_ids.index_select(-1, kept)
    pe = None if position_embeddings is None else tuple(x.index_select(-2, kept) for x in position_embeddings)
    return ids, pe


class GhostController:
    def __init__(self, model, router, cfg: GhostConfig, action="compact_route", model_family="llava"):
        if action not in ("compact_route", "compact_route_stagewise"):
            raise ValueError("Ghost extends only recoverable compact Reroute; no physical-delete Ghost baseline")
        if not isinstance(router, PDropRouter) or router.monotonic:
            raise ValueError("Use the original PDropRouter with monotonic=False")
        if model_family not in ("llava", "qwen25vl"):
            raise ValueError("Unsupported model family")
        if hasattr(model, "_cold_ghost_controller") or hasattr(model, "_routing_pre_hook_handle"):
            raise RuntimeError("Model already patched; unpatch it before installing Cold/Ghost")
        self.model, self.router, self.cfg = model, router, cfg
        self.layers = model.model.language_model.layers
        self.decisions = tuple(router.drop_layers)
        self.action, self.model_family = action, model_family
        hidden_dim = self.layers[0].self_attn.q_proj.in_features
        # Freeze BEFORE registering the new trainable module.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        devices = {layer.self_attn.q_proj.weight.device for layer in self.layers}
        if len(devices) != 1 or any(d.type == "meta" for d in devices):
            raise ValueError("Cold/Ghost currently requires all decoder layers on one device; disable CPU/offload or multi-device auto placement")
        self.bank = GhostBank(hidden_dim, len(self.layers), list(self.decisions), cfg).to(next(iter(devices)))
        model.add_module("_cold_ghost_bank", self.bank)
        model._cold_ghost_controller = self
        self.ctx = original.RoutingContext((0, 0), router, TokenDispatcher(), action)
        self.ctx._model_family = model_family
        self.ctx._layer_modules = dict(enumerate(self.layers))
        self.mode = "inference"
        self.teacher_states = {}
        self.capture_diagnostics = False
        self.diagnostics = []
        self.history = []
        self.reset((0, 0))
        self.original_forwards = [layer.forward for layer in self.layers]
        for i, layer in enumerate(self.layers):
            layer.forward = self._wrap(self.original_forwards[i], i)
        self.hook = model.register_forward_pre_hook(self._pre_forward, with_kwargs=True)

    def reset(self, visual_range):
        self.ctx.visual_token_range = visual_range
        self.router.reset()
        self.ctx.routing_log.clear()
        self.ctx.attn_weights_cache = None
        self.kept = self.deferred = self.deferred_hidden = None
        self.full_length = 0
        self.age = None
        self.decision = None
        self.ghost = None
        self.ghost_log = {}
        self.age_log = {}
        self.trace = TrainingTrace()
        self.diagnostics = []
        self.current_stats = {"ghost_layer_calls": 0, "ghost_token_updates": 0,
                              "reactivation_candidates": 0, "deferred_peak_bytes": 0}
        self.prefill = True
        for key in ("_stage_compact_kept_indices", "_stage_deferred_indices", "_stage_deferred_hidden"):
            setattr(self.ctx, key, None)
        self.ctx._stage_full_seq_len = 0

    def _pre_forward(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            if kwargs.get("inputs_embeds") is not None:
                raise ValueError("Provide input_ids for automatic visual range detection; inputs_embeds-only prefill is unsupported")
            return
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Original compact paths and Cold/Ghost require batch_size=1")
        cache = kwargs.get("past_key_values")
        has_past = cache is not None
        if has_past and hasattr(cache, "get_seq_length"):
            has_past = cache.get_seq_length() > 0
        # A decode call never resets or updates the deferred state.
        if ids.shape[-1] == 1 and has_past:
            self.prefill = False
            return
        if has_past:
            raise ValueError("Chunked/continued prefill is unsupported; start a fresh sample without KV cache")
        mask = kwargs.get("attention_mask")
        if mask is not None and mask.ndim == 2 and not bool(mask.bool().all()):
            raise ValueError("Padded batches are unsupported by the original compact protocol; use unpadded batch_size=1")
        finder = original.get_visual_token_finder(self.model_family)
        visual_range = finder(module, ids, pixel_values=kwargs.get("pixel_values"))
        if kwargs.get("pixel_values_videos") is not None:
            raise ValueError("This release covers single-image experiments, not video inputs")
        if self.model_family == "qwen25vl":
            count = (ids == module.config.vision_start_token_id).sum().item()
            if count > 1:
                raise ValueError("Only one contiguous image region per sample is supported")
        elif visual_range[1] > visual_range[0]:
            n = (ids == module.config.image_token_index).sum().item()
            positions = (ids[0] == module.config.image_token_index).nonzero(as_tuple=True)[0]
            expected = torch.arange(*visual_range, device=ids.device)
            if n != visual_range[1]-visual_range[0] or not torch.equal(positions, expected):
                raise ValueError("LLaVA must have exactly one processor-expanded contiguous image region")
        if visual_range[1] > ids.shape[-1]:
            raise ValueError("Invalid visual span: check processor/image token expansion")
        if self.current_stats["ghost_layer_calls"]:
            # Keep only small counters, never hidden tensors, across samples.
            self.history.append(dict(self.current_stats))
            self.history = self.history[-100:]
        self.reset(visual_range)
        if self.mode == "dense":
            self.teacher_states = {}

    @contextmanager
    def using(self, mode):
        if mode not in ("dense", "identity", "warmup", "rollout", "inference"):
            raise ValueError(mode)
        previous = self.mode
        self.mode = mode
        try:
            yield self
        finally:
            self.mode = previous

    def remove(self):
        self.hook.remove()
        for layer, forward in zip(self.layers, self.original_forwards):
            layer.forward = forward
        del self.model._cold_ghost_controller
        del self.model._cold_ghost_bank

    def _teacher(self, layer, indices, like):
        if layer not in self.teacher_states:
            raise RuntimeError("Run a dense teacher prefill on the SAME prepared input before each student sample")
        # Transfer only this layer; the dense trace stays detached in host RAM.
        value = self.teacher_states[layer].to(device=like.device, dtype=like.dtype)
        return value.index_select(1, indices)

    def _reconstruct(self, compact):
        full = compact.new_zeros((compact.shape[0], self.full_length, compact.shape[-1]))
        full = full.index_copy(1, self.deferred, self.deferred_hidden)
        return full.index_copy(1, self.kept, compact)

    def logical_hidden(self, hidden):
        if self.action == "compact_route_stagewise" and self.kept is not None:
            return self._reconstruct(hidden)
        return hidden

    def _partition(self, layer_idx, hidden):
        start, end = self.ctx.visual_token_range
        old_mask = None if self.decision is None else self.decision.selected_mask
        # No stale-score fallback. OOM or absent scoring fails this run explicitly.
        self.ctx.attn_weights_cache = None
        with torch.no_grad():
            original._capture_attention_weights(layer_idx, self.ctx, hidden, self.scoring_mask, self.scoring_pe)
        if self.ctx.attn_weights_cache is None:
            raise RuntimeError(f"Full attention scorer returned no scores at layer {layer_idx}")
        decision = self.router.compute_scores(layer_idx, self.ctx.attn_weights_cache, hidden, (start, end))
        self.ctx.attn_weights_cache = None  # quadratic scoring buffer must not live through the stage
        if not torch.isfinite(decision.scores).all():
            raise FloatingPointError("Non-finite original Full scores")
        self.decision = decision
        if self.mode in ("warmup", "rollout"):
            self.trace.full_masks[layer_idx] = decision.selected_mask.detach()
        if old_mask is not None and self.mode == "rollout":
            indices = (decision.selected_mask & ~old_mask)[0].nonzero(as_tuple=True)[0]
            if indices.numel():
                state = hidden[:, start:end].index_select(1, indices)
                self.trace.freshness.append((state, self._teacher(layer_idx, indices, state)))
        if old_mask is not None and self.capture_diagnostics:
            indices = (decision.selected_mask & ~old_mask)[0].nonzero(as_tuple=True)[0]
            if indices.numel():
                state = hidden[:, start:end].index_select(1, indices)
                target = self._teacher(layer_idx, indices, state)
                mse = relative_error(state, target, self.cfg.epsilon).detach().cpu()[0]
                cosine = (1-torch.nn.functional.cosine_similarity(state.float(), target.float(), dim=-1)).detach().cpu()[0]
                ages = self.age.index_select(1, indices).detach().cpu()[0]
                for token, age, m, c in zip(indices.tolist(), ages.tolist(), mse.tolist(), cosine.tolist()):
                    self.diagnostics.append(dict(layer=layer_idx, token=token, skip_age=age, relative_mse=m, cosine_error=c))
        candidates = (~decision.selected_mask)[0].nonzero(as_tuple=True)[0]
        self.ghost = candidates[:0]
        self.decision_z = self.decision_norm = None
        if layer_idx == self.decisions[-1] or candidates.numel() == 0 or not self.cfg.enabled or self.cfg.budget == 0 or self.mode == "identity":
            return
        block = self.bank.for_layer(layer_idx)
        self.decision_norm = self.layers[layer_idx].input_layernorm(hidden[:, start:end].index_select(1, candidates))
        scores = decision.scores.index_select(1, candidates)
        threshold = decision.scores.gather(1, decision.selected_indices).min(dim=-1, keepdim=True).values
        mass = decision.scores.sum(dim=-1, keepdim=True) + self.cfg.epsilon
        next_layer = self.decisions[self.decisions.index(layer_idx)+1]
        logits, z = block.reactivation_logits(self.decision_norm, scores, threshold, mass,
                                              self.age.index_select(1, candidates), (next_layer-layer_idx)/len(self.layers))
        self.current_stats["reactivation_candidates"] += candidates.numel()
        if self.mode in ("warmup", "rollout"):
            self.trace.reactivation[layer_idx] = (logits, candidates, next_layer)
        k = min(self.cfg.budget, candidates.numel())
        if self.cfg.ablation == "uniform_ghost":
            rows = torch.linspace(0, candidates.numel()-1, k, device=candidates.device).round().long()
        else:
            # Only Ghost tie-breaking is specified here. Original Full Top-K is untouched.
            rows = torch.argsort(logits[0].detach(), descending=True, stable=True)[:k]
        rows = rows.sort().values
        self.ghost = candidates.index_select(0, rows)
        if self.mode == "warmup":
            self.decision_z = z
        else:
            self.decision_z = z.index_select(1, rows)
            self.decision_norm = self.decision_norm.index_select(1, rows)

    def _wrap(self, forward, layer_idx):
        @functools.wraps(forward)
        def wrapped(hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                    use_cache=False, position_embeddings=None, **kwargs):
            call_kwargs = dict(kwargs, attention_mask=attention_mask, position_ids=position_ids,
                               past_key_values=past_key_values, use_cache=use_cache,
                               position_embeddings=position_embeddings)
            start, end = self.ctx.visual_token_range
            if not self.prefill or end <= start:
                if not self.prefill and self.router.should_route(layer_idx):
                    call_kwargs["attention_mask"] = None  # compact per-layer KV, one decode query
                return forward(hidden_states, **call_kwargs)
            if self.mode == "dense":
                self.teacher_states[layer_idx] = hidden_states[:, start:end].detach().to("cpu", copy=True)
                result = forward(hidden_states, **call_kwargs)
                self.teacher_states[layer_idx+1] = first_hidden(result)[:, start:end].detach().to("cpu", copy=True)
                return result
            if not self.router.should_route(layer_idx):
                return forward(hidden_states, **call_kwargs)
            if self.age is None:
                self.age = torch.zeros((1, end-start), dtype=torch.long, device=hidden_states.device)
            decision_layer = self.router.is_decision_layer(layer_idx)
            stagewise = self.action == "compact_route_stagewise"
            in_stage = stagewise and self.kept is not None
            full = self._reconstruct(hidden_states) if in_stage and decision_layer else hidden_states
            if decision_layer:
                self.scoring_mask, self.scoring_pe = attention_mask, position_embeddings
                self._partition(layer_idx, full)
                # Do not retain full-sequence graph via temporary scoring attributes.
                self.scoring_mask = self.scoring_pe = None
            decision = self.decision
            if decision is None:
                raise RuntimeError("Missing original cached routing decision")
            self.ctx.routing_log[layer_idx] = decision
            self.ghost_log[layer_idx] = self.ghost.detach().clone()
            if self.capture_diagnostics:
                self.age_log[layer_idx] = self.age.detach().cpu().clone()
            if not in_stage or decision_layer or not stagewise:
                self.full_length = full.shape[1]
                deferred = (~decision.selected_mask)[0].nonzero(as_tuple=True)[0] + start
                keep = torch.ones(self.full_length, device=full.device, dtype=torch.bool)
                keep[deferred] = False
                self.kept = keep.nonzero(as_tuple=True)[0]
                self.deferred = deferred
                self.deferred_hidden = full.index_select(1, deferred)
                compact = full.index_select(1, self.kept)
            else:
                compact = hidden_states
            visual_indices = decision.selected_indices[0]
            full_abs = visual_indices + start
            full_rows = torch.searchsorted(self.kept, full_abs)
            before = compact.index_select(1, full_rows)
            ids, pe = slice_positions(position_ids, position_embeddings, self.kept)
            call_kwargs.update(attention_mask=None, position_ids=ids, position_embeddings=pe)
            cache_position = call_kwargs.get("cache_position")
            if cache_position is not None and cache_position.shape[-1] == self.full_length:
                call_kwargs["cache_position"] = cache_position.index_select(-1, self.kept)
            result = forward(compact, **call_kwargs)
            compact_out = first_hidden(result)
            if layer_idx < self.decisions[-1] and self.cfg.enabled and self.cfg.budget > 0 and self.mode != "identity":
                update_indices = ((~decision.selected_mask)[0].nonzero(as_tuple=True)[0]
                                  if self.mode == "warmup" else self.ghost)
                if update_indices.numel():
                    deferred_rows = torch.searchsorted(self.deferred, update_indices+start)
                    skip = self.deferred_hidden.index_select(1, deferred_rows)
                    norm = (self.decision_norm if decision_layer and self.decision_norm is not None
                            else self.layers[layer_idx].input_layernorm(skip))
                    z = self.decision_z if decision_layer else None
                    observed = compact_out.index_select(1, full_rows)-before
                    block = self.bank.for_layer(layer_idx)
                    prediction = block.predict(norm, self.layers[layer_idx].input_layernorm(before), observed,
                                               self.age.index_select(1, update_indices), z=z)
                    if self.mode in ("warmup", "rollout"):
                        target = self._teacher(layer_idx+1, update_indices, prediction)-self._teacher(layer_idx, update_indices, prediction)
                        self.trace.residuals.append((prediction, target))
                    if self.mode != "warmup":
                        self.deferred_hidden = self.deferred_hidden.index_copy(1, deferred_rows, skip+prediction)
                    self.current_stats["ghost_layer_calls"] += 1
                    self.current_stats["ghost_token_updates"] += update_indices.numel()
            self.decision_z = self.decision_norm = None
            self.age = torch.where(decision.selected_mask, 0, self.age+1)
            nbytes = self.deferred_hidden.numel()*self.deferred_hidden.element_size()
            self.current_stats["deferred_peak_bytes"] = max(self.current_stats["deferred_peak_bytes"], nbytes)
            self.ctx._stage_compact_kept_indices = self.kept
            self.ctx._stage_deferred_indices = self.deferred
            self.ctx._stage_deferred_hidden = self.deferred_hidden
            self.ctx._stage_full_seq_len = self.full_length
            if stagewise:
                return with_hidden(result, compact_out)
            return with_hidden(result, self._reconstruct(compact_out))
        return wrapped


def install(model, router, cfg=None, action="compact_route", model_family="llava"):
    cfg = cfg if isinstance(cfg, GhostConfig) else GhostConfig.from_dict(cfg)
    if not cfg.enabled:
        return original.patch_model_for_routing(model, router, TokenDispatcher(), action, model_family)
    return GhostController(model, router, cfg, action, model_family)
