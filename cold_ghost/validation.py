"""Source immutability, syntax, environment and original lmms patch checks."""
import ast
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from .config import ROOT
from .checkpoints import file_sha256


def verify_source():
    manifest = json.loads((ROOT/".github/original-manifest.json").read_text())
    changes = [name for name, digest in manifest.items() if not (ROOT/name).is_file() or file_sha256(ROOT/name) != digest]
    if changes:
        raise RuntimeError(f"Original source content changed or missing: {changes}")
    syntax_errors, known_legacy, checked = [], [], 0
    for path in sorted(ROOT.rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        if any(part in (".git", "external", "__pycache__", ".venv") for part in path.parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
            checked += 1
        except SyntaxError as exc:
            item = dict(path=rel, line=exc.lineno, message=exc.msg)
            (known_legacy if rel == "test.py" and rel in manifest else syntax_errors).append(item)
    shell_count = 0
    for path in ROOT.rglob("*.sh"):
        if "external" in path.parts:
            continue
        subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
        shell_count += 1
    if syntax_errors:
        raise RuntimeError(f"New/runnable source syntax errors: {syntax_errors}")
    return dict(original_files=len(manifest), originals_byte_identical=True, python_files_passed=checked,
                shell_files_passed=shell_count, known_unmodified_legacy_syntax_errors=known_legacy,
                note="Original root test.py is a broken scratch example, not a runnable project entrypoint. See cold_ghost/examples/gqa_fixed.py.")


def doctor(require_hf=False, require_lmms=False, gpu=False):
    import torch
    report = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                  cuda_available=torch.cuda.is_available(), checks=verify_source(), dependencies={}, problems=[])
    for name in ("transformers", "accelerate", "hydra-core", "omegaconf", "lmms_eval", "torchvision", "pytest"):
        try:
            report["dependencies"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["dependencies"][name] = None
    if require_hf:
        for name in ("transformers", "accelerate", "hydra-core", "omegaconf"):
            if not report["dependencies"][name]:
                report["problems"].append(f"Missing {name}")
        if report["dependencies"]["transformers"] != "5.4.0":
            report["problems"].append("This release targets transformers==5.4.0; do not silently upgrade")
        try:
            from transformers import LlavaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:
            report["problems"].append(f"Transformers model import failed: {exc!r}")
    if require_lmms:
        try:
            package = importlib.import_module("lmms_eval")
            location = Path(package.__file__).parent
            if report["dependencies"]["lmms_eval"] != "0.7.1":
                report["problems"].append("lmms-eval must be the original v0.7.1 release")
            for task in ("refcoco", "refcoco+", "refcocog"):
                path = location/"tasks"/task/"utils_rec.py"
                text = path.read_text() if path.exists() else ""
                if "BBOX_COORD_FORMAT" not in text or "REFCOCO_PROMPT_STYLE" not in text:
                    report["problems"].append(f"Original RefCOCO patch missing: {path}")
        except Exception as exc:
            report["problems"].append(f"lmms-eval import failed: {exc!r}")
    if gpu:
        if not torch.cuda.is_available():
            report["problems"].append("CUDA unavailable")
        else:
            report["gpu"] = torch.cuda.get_device_name(0)
            report["bf16_supported"] = torch.cuda.is_bf16_supported()
            # User-invoked server preflight only; not executed by local delivery tests.
            x = torch.randn(32,32,device="cuda")
            assert torch.isfinite(x@x).all()
            torch.cuda.synchronize()
    report["ok"] = not report["problems"]
    return report
