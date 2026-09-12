"""REAL Transformers architectures, tiny random weights, synthetic pixels, CPU.

No pretrained checkpoint download, no benchmark accuracy test, no GPU claim.
"""
import copy
import pytest
import torch
pytest.importorskip("transformers", reason="Pinned Transformers unavailable in local offline container; required in GitHub CPU CI")
from transformers import LlavaConfig, LlavaForConditionalGeneration, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from models.router import PDropRouter
from models.dispatcher import TokenDispatcher
from models.patching import patch_model_for_routing
from cold_ghost.config import GhostConfig
from cold_ghost.integration import install
from cold_ghost.losses import compute_losses
from cold_ghost.profiling import walk_kv

pytestmark=pytest.mark.hf
torch.set_num_threads(1)


def native_model(family):
    text=dict(vocab_size=128,hidden_size=16,intermediate_size=32,num_hidden_layers=6,
              num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,
              bos_token_id=1,eos_token_id=127,pad_token_id=0)
    if family=="llava":
        text["model_type"]="llama"
        vision=dict(hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,
                    image_size=4,patch_size=2,projection_dim=16)
        cfg=LlavaConfig(text_config=text,vision_config=vision,image_token_index=99,
                        image_seq_length=4,vision_feature_layer=-1)
        cfg._attn_implementation="sdpa"
        model=LlavaForConditionalGeneration(cfg).eval()
        batch=dict(input_ids=torch.tensor([[1,99,99,99,99,2,3]]),pixel_values=torch.randn(1,3,4,4),attention_mask=torch.ones(1,7,dtype=torch.long))
    else:
        text["rope_parameters"]={"rope_type":"default","mrope_section":[1,1,2]}
        vision=dict(depth=1,hidden_size=16,intermediate_size=32,num_heads=2,in_channels=3,
                    patch_size=2,temporal_patch_size=1,spatial_merge_size=2,window_size=4,
                    out_hidden_size=16,fullatt_block_indexes=[0])
        cfg=Qwen2_5_VLConfig(text_config=text,vision_config=vision,image_token_id=99,video_token_id=98,
                            vision_start_token_id=100,vision_end_token_id=101)
        cfg._attn_implementation="sdpa"
        model=Qwen2_5_VLForConditionalGeneration(cfg).eval()
        batch=dict(input_ids=torch.tensor([[1,100,99,99,99,99,101,2,3]]),pixel_values=torch.randn(16,12),
                   image_grid_thw=torch.tensor([[1,4,4]]),attention_mask=torch.ones(1,9,dtype=torch.long))
    return model,batch


def patch(model,family,action):
    return install(model,PDropRouter([1,3,5],[.5,.5,.25],False),
                   GhostConfig(budget=1,bottleneck_dim=4,num_prototypes=2,block_share_span=2),action,family)


@pytest.mark.parametrize("family",["llava","qwen25vl"])
@pytest.mark.parametrize("action",["compact_route","compact_route_stagewise"])
def test_native_zero_init_original_equivalence(family,action):
    torch.manual_seed(42)
    original,batch=native_model(family)
    ghost=copy.deepcopy(original)
    patch_model_for_routing(original,PDropRouter([1,3,5],[.5,.5,.25],False),TokenDispatcher(),action,family)
    ctrl=patch(ghost,family,action)
    with torch.inference_mode():
        a=original(**batch,use_cache=True)
        b=ghost(**batch,use_cache=True)
    torch.testing.assert_close(a.logits[:,-1],b.logits[:,-1],atol=1e-6,rtol=1e-5)
    assert walk_kv(a)==walk_kv(b)
    assert ctrl.ghost_log[5].numel()==0


@pytest.mark.parametrize("family",["llava","qwen25vl"])
def test_native_nonzero_paths_generate_and_backward(family):
    torch.manual_seed(43)
    m1,batch=native_model(family);m2=copy.deepcopy(m1)
    c1=patch(m1,family,"compact_route");c2=patch(m2,family,"compact_route_stagewise")
    with torch.no_grad():
        for block in c1.bank.blocks.values():
            block.up.weight.normal_(std=.03);block.context_up.weight.normal_(std=.03)
    c2.bank.load_state_dict(c1.bank.state_dict())
    with torch.inference_mode():
        a=m1(**batch,use_cache=True);b=m2(**batch,use_cache=True)
        torch.testing.assert_close(a.logits[:,-1],b.logits[:,-1],atol=1e-6,rtol=1e-5)
        assert walk_kv(a)==walk_kv(b)
        ga=m1.generate(**batch,min_new_tokens=3,max_new_tokens=3,do_sample=False,use_cache=True)
        gb=m2.generate(**batch,min_new_tokens=3,max_new_tokens=3,do_sample=False,use_cache=True)
        assert torch.equal(ga,gb)
    assert c1.current_stats["ghost_layer_calls"]==4
    assert c2.current_stats["ghost_layer_calls"]==4
    with c1.using("dense"),torch.no_grad():m1(**batch,use_cache=False,logits_to_keep=1)
    with c1.using("rollout"):m1(**batch,use_cache=False,logits_to_keep=1)
    loss=compute_losses(c1.trace,c1.bank,"rollout")["total"]
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in c1.bank.parameters())
    assert all(p.grad is None for n,p in m1.named_parameters() if not n.startswith("_cold_ghost_bank"))
