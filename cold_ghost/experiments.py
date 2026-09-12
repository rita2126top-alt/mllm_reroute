"""Original task selectors plus complete additive experiment enumeration."""
from .config import ROOT

GROUNDING = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcoco+_val", "refcoco+_testA", "refcoco+_testB", "refcocog_val", "refcocog_test"]
TASKS = {
    "pope": ["pope"],
    "vqa": ["gqa", "mmbench", "mme"],
    "grounding": GROUNDING,
    "refcoco": GROUNDING[:3],
    "ablation": ["gqa", "mmbench", "refcoco_testA", "refcoco_testB"],
    "paper_main": ["paper_main"],
    "all": ["pope", "gqa", "mmbench", "mme", *GROUNDING],
}


def task_names(selector):
    names = TASKS.get(selector, selector.split(","))
    for name in names:
        if not (ROOT/"configs/eval"/f"{name}.yaml").is_file():
            raise ValueError(f"Unknown original eval config: {name}")
    return names


def benchmark_names():
    from omegaconf import OmegaConf
    names = []
    for task in TASKS["all"]:
        cfg = OmegaConf.load(ROOT/"configs/eval"/f"{task}.yaml")
        # Original eval group configs are the task body, not wrapped in eval.
        body = cfg.get("eval", cfg)
        names.extend(str(x) for x in body.benchmarks)
    return sorted(set(names))


def config_names(group="ghost", model="all", tier="all", training=False):
    if group not in ("all", "original", "ghost"):
        raise ValueError(group)
    if model not in ("all", "llava15", "qwen25vl"):
        raise ValueError(model)
    if tier not in ("all", "avg64", "avg128", "avg192"):
        raise ValueError(tier)
    results = []
    base = ROOT/"configs/experiment"
    for path in sorted(base.rglob("*.yaml")):
        is_ghost = "ghost" in path.relative_to(base).parts
        if group == "ghost" and not is_ghost or group == "original" and is_ghost:
            continue
        if training and (not is_ghost or "stagewise" in path.stem):
            continue
        if model != "all" and model not in path.stem:
            continue
        if tier != "all" and tier not in path.parts and "baseline" not in path.parts:
            continue
        results.append(str(path.relative_to(ROOT/"configs").with_suffix("")))
    return results
