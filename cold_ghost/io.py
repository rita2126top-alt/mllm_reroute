"""Model and processor loading for training/inference, using original model configs."""
from pathlib import Path
import torch
from PIL import Image
from .config import family, build_router, GhostConfig, identity, checkpoint_path
from .integration import install
from .checkpoints import load_checkpoint, file_sha256


def load_backbone(cfg, device="cuda:0"):
    from transformers import AutoProcessor, LlavaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
    model_family = family(cfg)
    dtype = torch.float16 if model_family == "llava" else torch.bfloat16
    if device == "cpu":
        dtype = torch.float32
    elif not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 7B training/evaluation. For local CPU checks use `python -m pytest`, not pretrained 7B loading.")
    cls = LlavaForConditionalGeneration if model_family == "llava" else Qwen2_5_VLForConditionalGeneration
    model = cls.from_pretrained(str(cfg.model.pretrained), dtype=dtype,
                                attn_implementation=str(cfg.model.get("attn_implementation", "sdpa")),
                                device_map={"": device}).eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    kwargs = {"max_pixels": int(cfg.model.max_pixels)} if cfg.model.get("max_pixels") else {}
    processor = AutoProcessor.from_pretrained(str(cfg.model.pretrained), **kwargs)
    return model, processor


def attach(cfg, model, *, training=False):
    router = build_router(cfg)
    if not cfg.get("ghost", {}).get("enabled", False):
        if router is not None:
            from models.patching import patch_model_for_routing
            from models.dispatcher import TokenDispatcher
            patch_model_for_routing(model, router, TokenDispatcher(), str(cfg.routing.action), family(cfg))
        return None
    ctrl = install(model, router, GhostConfig.from_dict(cfg.ghost), str(cfg.routing.action), family(cfg))
    ctrl.signature = identity(cfg, ctrl.bank.hidden_dim, ctrl.bank.num_layers)
    if not training:
        path = checkpoint_path(cfg)
        ctrl.checkpoint_metadata = load_checkpoint(path, ctrl.bank, ctrl.signature)
        ctrl.checkpoint_sha256 = file_sha256(path)
        ctrl.bank.eval()
    return ctrl


def prepare_inputs(model, processor, record):
    """Match the original benchmark's image+question chat template and numeric dtypes."""
    with Image.open(record["image_path"]) as image:
        image = image.convert("RGB")
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": record["question"]}]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        raw = processor(text=prompt, images=image, return_tensors="pt")
    parameter = next(model.parameters())
    inputs = {}
    for key, value in raw.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device=parameter.device, dtype=parameter.dtype if value.is_floating_point() else torch.long)
        else:
            inputs[key] = value
    return inputs
