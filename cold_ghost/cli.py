"""Run `python -m cold_ghost.cli --help` from the repository root."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
from .config import ROOT, read_config
from .experiments import config_names, task_names


def configuration(args):
    from .config import apply_overrides
    return apply_overrides(read_config(args.config), getattr(args, "checkpoint", None), getattr(args, "ablation", None))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    d=commands.add_parser("doctor", help="Check source integrity and dependencies without loading 7B weights")
    d.add_argument("--require-hf",action="store_true");d.add_argument("--require-lmms",action="store_true");d.add_argument("--gpu",action="store_true")
    commands.add_parser("verify", help="Verify all 79 original file hashes, Python ASTs and bash syntax")
    ls=commands.add_parser("list",help="List exact experiment configs")
    for obj in (ls,):
        obj.add_argument("--group",choices=["all","original","ghost"],default="all")
        obj.add_argument("--model",choices=["all","llava15","qwen25vl"],default="all")
        obj.add_argument("--tier",choices=["all","avg64","avg128","avg192"],default="all")
    sp=commands.add_parser("split-data",help="Split an independent source JSONL by image groups (seed 42)")
    sp.add_argument("--source",required=True);sp.add_argument("--train",default="data/ghost_train.jsonl");sp.add_argument("--val",default="data/ghost_val.jsonl")
    sp.add_argument("--val-fraction",type=float,default=.1)
    ix=commands.add_parser("index-eval",help="Download/enumerate original task images into an exclusion hash index")
    ix.add_argument("--out",default="data/eval_images.jsonl")
    au=commands.add_parser("audit-data",help="Reject train/val/test image leakage before training")
    tr=commands.add_parser("train",help="1000-update warm-up then 2000-update closed-loop rollout")
    for obj in (au,tr):
        obj.add_argument("--train",default="data/ghost_train.jsonl");obj.add_argument("--val",default="data/ghost_val.jsonl");obj.add_argument("--eval-index",default="data/eval_images.jsonl")
    tr.add_argument("--resume");tr.add_argument("--debug-updates",type=int,help="Engineering-only phase length; checkpoints are marked DEBUG and forbidden in benchmark evaluation")
    ev=commands.add_parser("eval",help="Original lmms-eval tasks and metrics with the trained adapter")
    ev.add_argument("--task",default="pope");ev.add_argument("--limit",type=int)
    pf=commands.add_parser("profile",help="Executed-op prefill FLOPs and original per-layer KV statistics")
    rt=commands.add_parser("runtime",help="Original CUDA Events prefill/decode runtime protocol")
    for obj,passes,warm in ((pf,1,0),(rt,5,2)):
        obj.add_argument("--passes",type=int,default=passes);obj.add_argument("--warmup",type=int,default=warm)
    rt.add_argument("--decode-tokens",type=int,default=64)
    inf=commands.add_parser("infer",help="Single-image greedy inference")
    inf.add_argument("--image",required=True);inf.add_argument("--question",required=True);inf.add_argument("--max-new-tokens",type=int,default=64)
    dg=commands.add_parser("diagnose",help="Compare reference/Ghost reactivation freshness and residual span")
    dg.add_argument("--manifest",default="data/ghost_val.jsonl");dg.add_argument("--limit",type=int,default=32)
    for obj in (tr,ev,pf,rt,inf,dg):
        obj.add_argument("--config",required=True,help="Path below configs, e.g. experiment/ghost/llava15/avg64/llava15_ghost_ours_vs_fastv_avg64")
        obj.add_argument("--checkpoint")
        obj.add_argument("--out",required=obj not in (tr,inf))
        obj.add_argument("--ablation",choices=["full","self_only","context_only","no_fresh","uniform_ghost"])
    sw=commands.add_parser("sweep",help="Run all selected configs; subprocess failure stops the sweep")
    sw.add_argument("--kind",choices=["train","eval","profile","runtime"],required=True)
    sw.add_argument("--group",choices=["all","original","ghost"],default="ghost")
    sw.add_argument("--model",choices=["all","llava15","qwen25vl"],default="all")
    sw.add_argument("--tier",choices=["all","avg64","avg128","avg192"],default="all")
    sw.add_argument("--tasks",default="all");sw.add_argument("--limit",type=int)
    sw.add_argument("--out",default="experiments/cold_ghost");sw.add_argument("--dry-run",action="store_true")
    sw.add_argument("--train",default="data/ghost_train.jsonl");sw.add_argument("--val",default="data/ghost_val.jsonl");sw.add_argument("--eval-index",default="data/eval_images.jsonl")
    collect=commands.add_parser("collect",help="Collect results without averaging incompatible metrics or FLOP backends")
    collect.add_argument("--root",default="experiments/cold_ghost");collect.add_argument("--out",default="experiments/cold_ghost/summary.csv")
    return p


def sweep_commands(args):
    names = config_names(args.group,args.model,args.tier,training=args.kind=="train")
    if not names:
        raise ValueError("No configurations selected")
    commands=[]
    for name in names:
        prefix=[sys.executable,"-m","cold_ghost.cli",args.kind,"--config",name]
        tag=name.replace("/","_")
        if args.kind=="train":
            commands.append(prefix+["--train",str(Path(args.train).resolve()),"--val",str(Path(args.val).resolve()),"--eval-index",str(Path(args.eval_index).resolve())])
        elif args.kind=="eval":
            for task in task_names(args.tasks):
                cmd=prefix+["--task",task,"--out",str(Path(args.out).resolve()/"eval"/tag/task)]
                if args.limit is not None:cmd += ["--limit",str(args.limit)]
                commands.append(cmd)
        else:
            commands.append(prefix+["--out",str(Path(args.out).resolve()/args.kind/f"{tag}.json")])
    return commands


def collect_results(root,out):
    rows=[]
    for path in sorted(Path(root).rglob("*.json")):
        if path.name=="results.json":
            row=json.loads(path.read_text());row["kind"]="eval"
            meta=path.parent/"run_manifest.json"
            if meta.exists():
                m=json.loads(meta.read_text());row["config"]=m["config"];row["limit"]=m.get("limit");row["checkpoint_sha256"]=m.get("ghost",{}).get("checkpoint_sha256")
        elif path.parent.name in ("profile","runtime"):
            data=json.loads(path.read_text())
            if "aggregate" not in data:continue
            row=dict(data["aggregate"],kind=path.parent.name,config=data.get("config"),
                     flops_backend=data.get("settings",{}).get("flops_backend"),checkpoint_sha256=data.get("ghost",{}).get("checkpoint_sha256"))
        else:continue
        row["artifact"]=str(path)
        rows.append({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    if not rows:raise ValueError(f"No completed results under {root}")
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for row in rows for k in row})
    with out.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return dict(rows=len(rows),output=str(out))


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        command=args.command
        if command=="doctor":
            from .validation import doctor
            result=doctor(args.require_hf,args.require_lmms,args.gpu)
            print(json.dumps(result,indent=2,ensure_ascii=False));return 0 if result["ok"] else 1
        if command=="verify":
            from .validation import verify_source
            result=verify_source()
        elif command=="list":
            names=config_names(args.group,args.model,args.tier)
            print("\n".join(names));print(f"# {len(names)} configurations");return 0
        elif command=="split-data":
            from .data import split_manifest
            result=split_manifest(args.source,args.train,args.val,args.val_fraction)
        elif command=="index-eval":
            from .data import index_evaluation_images
            result=index_evaluation_images(args.out)
        elif command=="audit-data":
            from .data import audit_data
            _,_,result=audit_data(args.train,args.val,args.eval_index)
        elif command=="train":
            from .training import train
            result=train(configuration(args),args.train,args.val,args.eval_index,args.out,args.resume,args.debug_updates)
        elif command=="eval":
            from .evaluation import evaluate
            result=evaluate(args.config,args.task,args.out,args.limit,args.checkpoint,args.ablation)
        elif command in ("profile","runtime"):
            from .profiling import profile,runtime
            if command=="profile":result=profile(args.config,args.out,args.passes,args.warmup,args.checkpoint,args.ablation)
            else:result=runtime(args.config,args.out,args.passes,args.warmup,args.decode_tokens,args.checkpoint,args.ablation)
        elif command=="infer":
            import torch
            from .io import load_backbone,attach,prepare_inputs
            cfg=configuration(args);model,processor=load_backbone(cfg);ctrl=attach(cfg,model)
            batch=prepare_inputs(model,processor,dict(image_path=args.image,question=args.question))
            with torch.inference_mode():
                tokens=model.generate(**batch,max_new_tokens=args.max_new_tokens,do_sample=False,use_cache=True)
            text=processor.batch_decode(tokens[:,batch["input_ids"].shape[-1]:],skip_special_tokens=True)[0]
            result=dict(answer=text)
            if args.out:
                Path(args.out).parent.mkdir(parents=True,exist_ok=True)
                Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
        elif command=="diagnose":
            from .diagnostics import diagnose
            result=diagnose(configuration(args),args.manifest,args.out,args.limit)
        elif command=="sweep":
            commands=sweep_commands(args)
            for command in commands:
                print(shlex.join(command),flush=True)
                if not args.dry_run:subprocess.run(command,cwd=ROOT,check=True)
            result=dict(commands=len(commands),dry_run=args.dry_run)
        elif command=="collect":
            result=collect_results(args.root,args.out)
        else:raise AssertionError(command)
        print(json.dumps(result,indent=2,ensure_ascii=False,default=str));return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}",file=sys.stderr)
        # Include chained failure context: critical for identifying training sample/API errors.
        import traceback
        traceback.print_exc()
        return 1


if __name__=="__main__":
    raise SystemExit(main())
