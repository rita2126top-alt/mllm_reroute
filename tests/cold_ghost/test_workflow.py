import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from omegaconf import OmegaConf
from cold_ghost.config import ROOT, read_config, identity, GhostConfig
from cold_ghost.checkpoints import save_checkpoint, load_checkpoint, file_sha256
from cold_ghost.data import canonical_image_hash, read_manifest, audit_data, split_manifest
from cold_ghost.experiments import config_names, benchmark_names, task_names
from cold_ghost.training import train_loop, SampleOrder
from cold_ghost.profiling import walk_kv
from cold_ghost.cli import parser, sweep_commands
from cold_ghost.validation import verify_source
from .toy import ToyModel, inputs
from .test_core import controller

CFG="experiment/ghost/llava15/avg64/llava15_ghost_ours_vs_fastv_avg64"


def test_all_original_hashes_and_entrypoint_syntax():
    report=verify_source()
    assert report["original_files"]==79
    assert report["originals_byte_identical"]
    assert [r["path"] for r in report["known_unmodified_legacy_syntax_errors"]]==["test.py"]


def test_all_24_configs_preserve_original_routing_and_eval():
    assert len(config_names("original"))==38
    assert len(config_names("ghost"))==24
    assert len(config_names("ghost",training=True))==12
    for name in config_names("ghost"):
        original=name.replace("experiment/ghost/","experiment/").replace("_ghost_ours_","_ours_")
        a,b=read_config(name),read_config(original)
        assert OmegaConf.to_container(a.routing)==OmegaConf.to_container(b.routing)
        assert OmegaConf.to_container(a.model)==OmegaConf.to_container(b.model)
        assert OmegaConf.to_container(a.eval)==OmegaConf.to_container(b.eval)
        assert GhostConfig.from_dict(a.ghost)==GhostConfig()
        if "stagewise" in name:
            other=read_config(name.replace("_stagewise", ""))
            assert a.ghost.checkpoint==other.ghost.checkpoint
            assert identity(a,16,6)==identity(other,16,6)


def test_sweep_counts_and_task_selectors():
    args=parser().parse_args(["sweep","--kind","eval","--group","all","--dry-run"])
    assert len(sweep_commands(args))==62*12
    args.kind="train"
    assert len(sweep_commands(args))==12
    assert len(benchmark_names())==12
    assert task_names("paper_main")==["paper_main"]
    assert len(task_names("grounding"))==8
    assert task_names("vqa")==["gqa","mmbench","mme"]


def test_checkpoints_strict_identity_and_debug_guard(tmp_path):
    c=controller(ToyModel())
    signature=identity(read_config(CFG),16,6)
    p=tmp_path/"checkpoint.pt"
    save_checkpoint(p,c.bank,signature,phase="rollout",step=100,validated=True,warmup_completed=1000)
    load_checkpoint(p,c.bank,signature)
    wrong=copy.deepcopy(signature);wrong["routing"]["keep_ratios"][0]=.8
    with pytest.raises(ValueError,match="identity mismatch"):load_checkpoint(p,c.bank,wrong)
    save_checkpoint(p,c.bank,signature,phase="rollout",step=1,validated=True,debug=True,warmup_completed=1000)
    with pytest.raises(ValueError,match="non-debug"):load_checkpoint(p,c.bank,signature)
    with pytest.raises(FileNotFoundError,match="random-weight"):load_checkpoint(tmp_path/"missing.pt",c.bank,signature)


def make_data(tmp_path):
    for i in range(4):
        Image.new("RGB",(3,3),(i*50,1,2)).save(tmp_path/f"{i}.png")
    def manifest(name,indices):
        p=tmp_path/name
        p.write_text("".join(json.dumps(dict(sample_id=f"s{i}",image_id=f"image{i}",image_path=f"{i}.png",question="Question?"))+"\n" for i in indices))
        return p
    train=manifest("train.jsonl",[0,1]);val=manifest("val.jsonl",[2])
    index=tmp_path/"eval.jsonl"
    index.write_text(json.dumps(dict(sha256_rgb=canonical_image_hash(tmp_path/"3.png"),image_ids=["image3"]))+"\n")
    index.with_suffix(".meta.json").write_text(json.dumps(dict(complete=True,index_sha256=file_sha256(index),requested_tasks=benchmark_names())))
    return train,val,index,manifest


def test_data_audit_and_hash_leakage(tmp_path):
    train,val,index,manifest=make_data(tmp_path)
    a,b,report=audit_data(train,val,index)
    assert report["train_samples"]==2 and report["val_samples"]==1
    val=manifest("val.jsonl",[0])
    with pytest.raises(ValueError,match="Image leakage"):audit_data(train,val,index)
    val=manifest("val.jsonl",[3])
    with pytest.raises(ValueError,match="evaluation/bench"):audit_data(train,val,index)


def test_hash_ignores_filename_and_lossless_encoding(tmp_path):
    image=Image.new("RGB",(4,5),(21,22,23))
    image.save(tmp_path/"a.png");image.save(tmp_path/"b.bmp")
    assert canonical_image_hash(tmp_path/"a.png")==canonical_image_hash(tmp_path/"b.bmp")


def test_deterministic_image_group_split(tmp_path):
    train,val,index,manifest=make_data(tmp_path)
    source=manifest("source.jsonl",[0,1,2,3])
    a,b=tmp_path/"newtrain.jsonl",tmp_path/"newval.jsonl"
    split_manifest(source,a,b)
    ar,br=read_manifest(a),read_manifest(b)
    assert len(ar)+len(br)==4
    assert not {r["image_id"] for r in ar}&{r["image_id"] for r in br}
    with pytest.raises(FileExistsError):split_manifest(source,a,b)


def test_partial_or_modified_eval_index_is_rejected(tmp_path):
    train,val,index,_=make_data(tmp_path)
    index.write_text(index.read_text()+"\n")
    with pytest.raises(ValueError,match="partial, modified"):audit_data(train,val,index)


def test_tiny_two_phase_training_and_resume(tmp_path):
    torch.manual_seed(42)
    m=ToyModel();c=controller(m)
    c.signature=identity(read_config(CFG),16,6)
    rows=[dict(sample_id="tiny-cpu-only")]
    summary=train_loop(m,c,rows,rows,lambda row:inputs(),tmp_path,{"test_fixture":True},
                       warmup_updates=2,rollout_updates=2,accumulation=2,validation_every=1,debug=True)
    assert summary["completed"]
    for name in ("best.pt","last.pt","warmup.pt","training.jsonl","training_summary.json"):
        assert (tmp_path/name).is_file()
    last=load_checkpoint(tmp_path/"last.pt",c.bank,c.signature,for_evaluation=False)
    assert last["training"]["phase"]=="rollout" and last["training"]["step"]==2
    assert last["optimizer"]["param_groups"][0]["lr"]==3e-5
    with pytest.raises(ValueError,match="non-debug"):
        load_checkpoint(tmp_path/"best.pt",c.bank,c.signature)
    # Resume at a completed phase is a no-op on weights, not a retraining/reset.
    weights={k:v.clone() for k,v in c.bank.state_dict().items()}
    train_loop(m,c,rows,rows,lambda row:inputs(),tmp_path,{"test_fixture":True},resume=tmp_path/"last.pt",
               warmup_updates=2,rollout_updates=2,accumulation=2,validation_every=1,debug=True)
    for key,value in c.bank.state_dict().items():assert torch.equal(value,weights[key])


def test_sampler_resume_offsets():
    a,b=SampleOrder(11),SampleOrder(11)
    assert [a.at(i) for i in range(16,35)]==[b.at(i) for i in range(16,35)]
    assert sorted(a.at(i) for i in range(11))==list(range(11))


def test_new_and_old_kv_formats():
    k=torch.zeros(1,2,3,4);v=torch.zeros_like(k)
    caches=[SimpleNamespace(layers=[SimpleNamespace(keys=k,values=v)]),SimpleNamespace(key_cache=[k],value_cache=[v]),[(k,v)]]
    for cache in caches:
        assert walk_kv(SimpleNamespace(past_key_values=cache))==(6,192,[3])
    with pytest.raises(RuntimeError):walk_kv(SimpleNamespace(past_key_values=None))


def test_bfloat16_forward_keeps_fp32_trainable_parameters():
    torch.manual_seed(42)
    m=ToyModel().to(torch.bfloat16);c=controller(m)
    with c.using("dense"),torch.no_grad():m(**inputs())
    from cold_ghost.training import sample_loss
    losses=sample_loss(m,c,inputs(),"warmup")
    losses["total"].backward()
    assert all(p.dtype==torch.float32 for p in c.bank.parameters())
    assert torch.isfinite(losses["total"])


def test_executed_flop_counter_includes_ghost():
    from torch.utils.flop_counter import FlopCounterMode
    m=ToyModel();c=controller(m)
    with c.using("identity"),FlopCounterMode(display=False) as a,torch.inference_mode():m(**inputs())
    with c.using("inference"),FlopCounterMode(display=False) as b,torch.inference_mode():m(**inputs())
    assert b.get_total_flops()>a.get_total_flops()>0
