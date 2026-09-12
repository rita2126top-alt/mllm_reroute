import copy
import pytest
import torch
from models.router import PDropRouter
from models.dispatcher import TokenDispatcher
from models.patching import patch_model_for_routing
from cold_ghost.config import GhostConfig
from cold_ghost.integration import install, slice_positions
from cold_ghost.losses import compute_losses, TrainingTrace
from .toy import ToyModel, inputs

torch.set_num_threads(1)

def controller(model, action="compact_route", budget=1):
    return install(model,PDropRouter([1,3,5],[.5,.5,.25],False),
                   GhostConfig(budget=budget,bottleneck_dim=4,num_prototypes=2,block_share_span=2),action)

def nonzero(c):
    with torch.no_grad():
        for block in c.bank.blocks.values():
            block.up.weight.normal_(std=.05)
            block.context_up.weight.normal_(std=.05)

@pytest.mark.parametrize("action",["compact_route","compact_route_stagewise"])
@pytest.mark.parametrize("tuple_output",[False,True])
def test_zero_init_matches_original(action,tuple_output):
    torch.manual_seed(42)
    base=ToyModel(tuple_output=tuple_output)
    m=copy.deepcopy(base)
    ref=patch_model_for_routing(base,PDropRouter([1,3,5],[.5,.5,.25],False),TokenDispatcher(),action)
    c=controller(m,action)
    with torch.no_grad():
        a,b=base(**inputs()),m(**inputs())
    torch.testing.assert_close(a.logits,b.logits,atol=0,rtol=0)
    for l in ref.routing_log:
        assert torch.equal(ref.routing_log[l].selected_mask,c.ctx.routing_log[l].selected_mask)
    assert c.ghost_log[5].numel()==0
    assert c.current_stats["ghost_layer_calls"]==4

@pytest.mark.parametrize("budget",[0,1,128])
def test_paths_equivalent_with_real_ghost(budget):
    torch.manual_seed(3)
    m1,m2=ToyModel(),ToyModel()
    m2.load_state_dict(m1.state_dict())
    c1,c2=controller(m1,budget=budget),controller(m2,"compact_route_stagewise",budget)
    nonzero(c1)
    c2.bank.load_state_dict(c1.bank.state_dict())
    with torch.no_grad():
        a,b=m1(**inputs(),use_cache=True),m2(**inputs(),use_cache=True)
    torch.testing.assert_close(a.logits[:,-1],b.logits[:,-1],atol=0,rtol=0)
    torch.testing.assert_close(a.last_hidden_state,c2.logical_hidden(b.last_hidden_state),atol=0,rtol=0)
    for l in c1.ghost_log:
        assert torch.equal(c1.ghost_log[l],c2.ghost_log[l])
    assert [k.shape[-2] for k in a.past_key_values.keys]==[7,5,5,5,5,4]
    before=c1.current_stats["ghost_layer_calls"]
    with torch.no_grad():
        da=m1(input_ids=torch.tensor([[4]]),past_key_values=a.past_key_values,use_cache=True)
        db=m2(input_ids=torch.tensor([[4]]),past_key_values=b.past_key_values,use_cache=True)
    torch.testing.assert_close(da.logits,db.logits,atol=0,rtol=0)
    assert c1.current_stats["ghost_layer_calls"]==before

@pytest.mark.parametrize("mode",["warmup","rollout"])
def test_training_gradients(mode):
    torch.manual_seed(8)
    m=ToyModel()
    c=controller(m)
    nonzero(c)
    with c.using("dense"),torch.no_grad():
        m(**inputs(),use_cache=False)
    assert len(c.teacher_states)==7
    with c.using(mode):
        m(**inputs(),use_cache=False)
    losses=compute_losses(c.trace,c.bank,mode)
    losses["total"].backward()
    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith("_cold_ghost_bank"))
    assert sum(p.grad.abs().sum().item() for p in c.bank.parameters() if p.grad is not None)>0
    assert set(c.trace.reactivation)=={1,3}
    assert set(c.trace.full_masks)=={1,3,5}
    assert torch.isfinite(losses["total"])


def test_cold_identity_and_ghost_writeback():
    torch.manual_seed(7)
    m=ToyModel()
    c=controller(m)
    nonzero(c)
    seen={}
    def hook(module,args,output):
        if module.i==1:
            x=args[0]
            y=output[0] if isinstance(output,tuple) else output
            d=c.ctx.routing_log[1]
            cold=(~d.selected_mask)[0].clone()
            cold[c.ghost_log[1]]=False
            cold=cold.nonzero(as_tuple=True)[0]+1
            torch.testing.assert_close(x.index_select(1,cold),y.index_select(1,cold),atol=0,rtol=0)
            g=c.ghost_log[1]+1
            assert not torch.equal(x.index_select(1,g),y.index_select(1,g))
            seen["ok"]=True
    handle=m.model.language_model.layers[1].register_forward_hook(hook)
    m(**inputs())
    handle.remove()
    assert seen["ok"]


def test_reset_text_and_image_and_age():
    m=ToyModel();c=controller(m)
    c.capture_diagnostics=False
    m(**inputs())
    assert c.age.max()>=1
    m(input_ids=torch.tensor([[1,2,3]]))
    assert c.ctx.visual_token_range==(0,0)
    assert c.age is None and not c.ctx.routing_log
    a=m(**inputs()).logits
    b=m(**inputs()).logits
    torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_freshness_backprop_to_earlier_ghost(monkeypatch):
    import models.patching as patching
    m=ToyModel();c=controller(m,budget=128)
    nonzero(c)
    def capture(layer,ctx,hidden,mask,pe):
        att=hidden.new_ones((1,2,7,7))
        # Decision 1 keeps 0,1; decision 3 keeps 2,3; final keeps 0.
        scores=[9,8,1,0] if layer in (1,5) else [0,1,8,9]
        att[:,:,-1,1:5]=torch.tensor(scores,dtype=hidden.dtype)
        ctx.attn_weights_cache=att
    monkeypatch.setattr(patching,"_capture_attention_weights",capture)
    with c.using("dense"),torch.no_grad():m(**inputs())
    with c.using("rollout"):m(**inputs())
    losses=compute_losses(c.trace,c.bank,"rollout")
    assert c.trace.freshness
    losses["freshness"].backward()
    assert c.bank.for_layer(1).up.weight.grad.abs().sum()>0


def test_empty_losses_differentiable_and_finite():
    c=controller(ToyModel())
    losses=compute_losses(TrainingTrace(),c.bank,"rollout")
    assert losses["total"].item()==0
    losses["total"].backward()


def test_no_stale_scorer_fallback(monkeypatch):
    import models.patching as patching
    c=controller(ToyModel())
    monkeypatch.setattr(patching,"_capture_attention_weights",lambda *args:None)
    with pytest.raises(RuntimeError,match="no scores"):
        c.model(**inputs())

@pytest.mark.parametrize("shape",[(1,7),(3,1,7)])
def test_positions_preserved(shape):
    ids=torch.arange(7).expand(shape)
    kept=torch.tensor([0,2,6])
    pe=torch.randn((1,7,8) if len(shape)==2 else (3,1,7,8))
    oi,op=slice_positions(ids,(pe,pe),kept)
    assert torch.equal(oi,ids[...,kept])
    assert torch.equal(op[0],pe[...,kept,:])


def test_reject_batch_padding_and_double_install():
    c=controller(ToyModel())
    with pytest.raises(ValueError,match="batch_size"):
        c.model(input_ids=torch.ones(2,7,dtype=torch.long))
    with pytest.raises(ValueError,match="Padded"):
        c.model(**inputs(),attention_mask=torch.tensor([[0,1,1,1,1,1,1]]))
    with pytest.raises(RuntimeError,match="already patched"):
        controller(c.model)


def test_sharing_and_parameter_initialization():
    c=controller(ToyModel())
    assert c.bank.for_layer(2) is c.bank.for_layer(3)
    b=c.bank.for_layer(1)
    assert torch.count_nonzero(b.up.weight)==0
    assert b.gate.bias.item()==-2
    assert b.reactivate.in_features==8
    with pytest.raises(ValueError):c.bank.for_layer(5)
