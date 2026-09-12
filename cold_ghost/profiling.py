"""Original cohorts/protocols; explicit FLOP counter provenance and Ghost costs."""
from __future__ import annotations
import json
import statistics
import sys
from pathlib import Path
import torch
from .config import ROOT, read_config, family, checkpoint_path, apply_overrides
from .io import load_backbone, attach
from .evaluation import metadata, temporary_attribute


def walk_kv(out):
    kv = out.past_key_values
    if kv is None:
        raise RuntimeError("Profiling requested use_cache=True but the model returned no KV cache")
    if hasattr(kv, "layers"):
        pairs = [(layer.keys, layer.values) for layer in kv.layers]
    elif hasattr(kv, "key_cache"):
        pairs = list(zip(kv.key_cache, kv.value_cache))
    else:
        pairs = [(item[0], item[1]) for item in kv]
    pairs = [(k,v) for k,v in pairs if k is not None and v is not None]
    if not pairs:
        raise RuntimeError("Unrecognized/empty KV cache; do not report a false zero-memory measurement")
    per_layer = [int(k.shape[-2]) for k,_ in pairs]
    tokens = sum(k.shape[-2]+v.shape[-2] for k,v in pairs)
    nbytes = sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in pairs)
    return int(tokens), int(nbytes), per_layer


def _load(name, checkpoint=None, ablation=None):
    cfg = read_config(name)
    apply_overrides(cfg, checkpoint, ablation)
    if cfg.get("ghost", {}).get("enabled", False) and not checkpoint_path(cfg).is_file():
        raise FileNotFoundError(f"Missing trained adapter: {checkpoint_path(cfg)}")
    model, processor = load_backbone(cfg)
    ctrl = attach(cfg, model)
    return model, processor, ctrl, cfg


def profile(name, out, passes=1, warmup=0, checkpoint=None, ablation=None):
    """Count supported executed ops, INCLUDING scorer and every Ghost projection.

    FlopCounterMode counts multiplication+addition as two FLOPs. Elementwise
    normalization/nonlinearities are not exhaustively counted; this is NOT an
    assertion of exact hardware FLOPs or equality to old DeepSpeed counts.
    Use this same entrypoint for original and Ghost comparison arms.
    """
    from torch.utils.flop_counter import FlopCounterMode
    from profiler.bench_runtime import _load_bundled_cohort, _build_inputs_for_sample
    if passes < 1 or warmup < 0:
        raise ValueError("passes>=1 and warmup>=0 required")
    model, processor, ctrl, cfg = _load(name, checkpoint, ablation)
    bench_dir, manifest = _load_bundled_cohort()
    prepared = [_build_inputs_for_sample(processor, model, row, bench_dir) for row in manifest]
    for _ in range(warmup):
        for batch in prepared:
            with torch.inference_mode():
                result = model(**batch, use_cache=True)
                del result
    records = []
    for row in manifest:
        records.append(dict(sample_id=row["sample_id"], image=row["image"], tflops=[], kv_cache_tokens=[], kv_cache_bytes=[], ghost=[], peak_allocated_bytes=[]))
    for _ in range(passes):
        for i, batch in enumerate(prepared):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            with FlopCounterMode(display=False) as counter, torch.inference_mode():
                result = model(**batch, use_cache=True)
            torch.cuda.synchronize()
            tokens, nbytes, layers = walk_kv(result)
            rec = records[i]
            rec["tflops"].append(counter.get_total_flops()/1e12)
            rec["kv_cache_tokens"].append(tokens)
            rec["kv_cache_bytes"].append(nbytes)
            rec["per_layer_kv_seq_len"] = layers
            rec["peak_allocated_bytes"].append(torch.cuda.max_memory_allocated())
            rec["ghost"].append(dict(ctrl.current_stats) if ctrl else {})
            del result
            torch.cuda.empty_cache()
    for rec in records:
        # Original profiler uses the upper median, including when the count is even.
        upper = lambda xs: sorted(xs)[len(xs)//2]
        rec["tflops_median"] = upper(rec["tflops"])
        rec["kv_cache_tokens_median"] = upper(rec["kv_cache_tokens"])
        rec["kv_cache_MB_median"] = upper(rec["kv_cache_bytes"])/2**20
    all_values = lambda key: [x for row in records for x in row[key]]
    summary = dict(config=name, scope="prefill_only", bench_data=dict(cohort_size=len(manifest), cohort_dir=str(bench_dir)),
                   settings=dict(n_passes=passes, n_warmup_passes=warmup, flops_backend="torch.utils.flop_counter.FlopCounterMode",
                                 flops_convention="2 FLOPs per multiply-add; supported matmul/conv/attention ops only; includes Full scoring and Ghost",
                                 comparable_to_legacy_deepspeed_without_recount=False),
                   per_sample=records, ghost=metadata(ctrl),
                   aggregate=dict(prefill_tflops_mean=statistics.mean(all_values("tflops")),
                                  prefill_kv_cache_tokens_mean=statistics.mean(all_values("kv_cache_tokens")),
                                  prefill_kv_cache_MB_mean=statistics.mean(all_values("kv_cache_bytes"))/2**20))
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2)+"\n")
    return summary["aggregate"]


def runtime(name, out, passes=5, warmup=2, decode_tokens=64, checkpoint=None, ablation=None):
    """Run the original CUDA Events timing loop without copying/redefining it."""
    from profiler import bench_runtime as upstream
    if passes < 1 or warmup < 0 or decode_tokens < 1:
        raise ValueError("Invalid runtime protocol arguments")
    holder = {}
    def loader(path):
        model, processor, ctrl, cfg = _load(name, checkpoint, ablation)
        holder["ctrl"] = ctrl
        return model, processor, family(cfg)
    old_argv = sys.argv
    sys.argv = ["bench_runtime.py", "--config-name", name, "--out", str(out),
                "--n-passes", str(passes), "--n-warmup-passes", str(warmup), "--n-decode-tokens", str(decode_tokens)]
    try:
        with temporary_attribute(upstream, "_build_model_and_apply_routing", loader):
            result = upstream.main()
        if result not in (None, 0):
            raise RuntimeError(f"Original runtime benchmark exited with {result}")
    finally:
        sys.argv = old_argv
    path = Path(out)
    summary = json.loads(path.read_text())
    summary["ghost"] = metadata(holder.get("ctrl"))
    path.write_text(json.dumps(summary, indent=2)+"\n")
    return summary.get("aggregate", {})
