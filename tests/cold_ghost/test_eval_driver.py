"""CPU integration of original evaluator plumbing with a synthetic harness.

No real benchmark metrics or pretrained Ghost weights are produced by tests.
"""
import json
import sys
import types
from pathlib import Path
import pytest
import torch
pytest.importorskip("hydra",reason="Hydra is installed and required in CPU CI")
from omegaconf import OmegaConf
from cold_ghost import evaluation
from cold_ghost.config import read_config, identity
from cold_ghost.experiments import config_names
from cold_ghost.io import attach
from cold_ghost.checkpoints import save_checkpoint
from .toy import ToyModel,inputs


def test_hydra_composition_of_all_added_configs():
    for name in config_names("ghost"):
        cfg=evaluation.compose_eval(name,"gqa",limit=2)
        original=read_config(name)
        assert OmegaConf.to_container(cfg.routing)==OmegaConf.to_container(original.routing)
        assert cfg.eval.benchmarks==["gqa"] and cfg.eval.limit==2
        assert cfg.ghost.budget==128


def test_original_eval_harness_hooks_and_checkpoint(tmp_path,monkeypatch):
    cfg=read_config(config_names("ghost",model="llava15")[0])
    cfg.routing.drop_layers=[1,3,5];cfg.routing.keep_ratios=[.5,.5,.25]
    cfg.ghost.bottleneck_dim=4;cfg.ghost.num_prototypes=2;cfg.ghost.block_share_span=2;cfg.ghost.budget=1
    cfg.ghost.checkpoint=str(tmp_path/"test-fixture.pt")
    cfg.eval.benchmarks=["pope"]
    fixture=ToyModel();c=attach(cfg,fixture,training=True)
    # This temporary fixture only tests serialization/harness contracts.
    save_checkpoint(cfg.ghost.checkpoint,c.bank,c.signature,phase="rollout",step=100,validated=True,warmup_completed=1000)
    class Wrapper:
        @classmethod
        def create_from_arg_string(cls,*args):
            return types.SimpleNamespace(_model=ToyModel())
    seen={}
    def simple_evaluate(**kwargs):
        model=kwargs["model"]._model
        assert hasattr(model,"_cold_ghost_controller")
        with torch.inference_mode():model(**inputs())
        seen["tasks"]=kwargs["tasks"]
        return {"results":{"pope":{"acc,none":.5}},"samples":{"pope":[]}}
    package=types.ModuleType("lmms_eval");package.evaluator=types.SimpleNamespace(simple_evaluate=simple_evaluate)
    models=types.ModuleType("lmms_eval.models");models.get_model=lambda name:Wrapper
    monkeypatch.setitem(sys.modules,"lmms_eval",package);monkeypatch.setitem(sys.modules,"lmms_eval.models",models)
    monkeypatch.setattr(evaluation,"compose_eval",lambda *args:cfg)
    from scripts import run_eval as upstream
    previous=upstream.patch_model_for_routing
    # Keep the test's synthetic global metric JSONL outside the source workspace.
    monkeypatch.setattr(upstream,"_PROJECT_ROOT",tmp_path)
    (tmp_path/"experiments").mkdir()
    result=evaluation.evaluate("synthetic-contract-test","pope",tmp_path/"out")
    assert result["status"]=="completed" and result["ghost"]["enabled"]
    assert seen["tasks"]==["pope"]
    assert upstream.patch_model_for_routing is previous
    assert (tmp_path/"out/results.json").exists()
    assert json.loads((tmp_path/"out/run_manifest.json").read_text())["ghost"]["checkpoint_sha256"]
