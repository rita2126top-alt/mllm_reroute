"""CPU control-flow test of FP16 overflow retry; does not claim CUDA arithmetic."""
from types import SimpleNamespace
import pytest
import torch
from cold_ghost.training import accumulated_update


class FakeScaler:
    def __init__(self, enabled=True):
        self.enabled=enabled;self.value=65536.;self.calls=0;self.steps=0
    def scale(self,loss):return loss
    def unscale_(self,optimizer):
        self.calls+=1
        if self.calls==1:optimizer.param_groups[0]["params"][0].grad.fill_(float("inf"))
    def is_enabled(self):return self.enabled
    def get_scale(self):return self.value
    def update(self,new_scale=None):
        if new_scale is not None:self.value=new_scale
    def step(self,optimizer):self.steps+=1;optimizer.step()


def setup():
    bank=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():bank.weight.fill_(.5)
    ctrl=SimpleNamespace(bank=bank,ctx=SimpleNamespace())
    optimizer=torch.optim.SGD(bank.parameters(),lr=.1)
    visited=[]
    def loss_fn(row):
        visited.append(row["sample_id"])
        loss=bank.weight.square().sum()
        return {key:loss for key in ("delta","direction","reactivation","freshness","total")}
    return ctrl,optimizer,visited,loss_fn


def test_overflow_retries_same_samples_without_extra_update():
    ctrl,optimizer,visited,loss_fn=setup();scaler=FakeScaler()
    _,_,retries=accumulated_update(ctrl,[{"sample_id":"a"},{"sample_id":"b"}],loss_fn,optimizer,scaler)
    assert retries==1 and scaler.steps==1 and scaler.get_scale()==32768
    assert visited==["a","b","a","b"]
    torch.testing.assert_close(ctrl.bank.weight,torch.tensor([[.4]]))


def test_without_scaler_nonfinite_gradients_fail_before_update():
    ctrl,optimizer,_,loss_fn=setup();scaler=FakeScaler(enabled=False)
    with pytest.raises(FloatingPointError):
        accumulated_update(ctrl,[{"sample_id":"a"}],loss_fn,optimizer,scaler)
    assert scaler.steps==0
    torch.testing.assert_close(ctrl.bank.weight,torch.tensor([[.5]]))
