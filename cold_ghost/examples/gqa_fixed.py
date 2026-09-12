"""Corrected standalone GQA/RoPE example; NOT part of the Cold/Ghost method.

The broken user-supplied root test.py is intentionally left byte-identical.
Run this example with: python -m cold_ghost.examples.gqa_fixed
"""
import math
import torch
from torch import nn


def apply_rope(x, positions):
    if x.shape[-1] % 2:
        raise ValueError("RoPE head_dim must be even")
    inv = 10000**(-torch.arange(0,x.shape[-1],2,device=x.device).float()/x.shape[-1])
    angles = positions[...,None].float()*inv
    cos,sin=angles.cos().unsqueeze(1).to(x.dtype),angles.sin().unsqueeze(1).to(x.dtype)
    even,odd=x[...,0::2],x[...,1::2]
    return torch.stack((even*cos-odd*sin,even*sin+odd*cos),dim=-1).flatten(-2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_q_heads, num_kv_heads):
        super().__init__()
        if hidden_dim%num_q_heads or num_q_heads%num_kv_heads:
            raise ValueError("hidden_dim must divide into Q heads; Q heads must divide into KV groups")
        self.head_dim=hidden_dim//num_q_heads
        if self.head_dim%2:raise ValueError("head_dim must be even")
        self.num_q_heads,self.num_kv_heads=num_q_heads,num_kv_heads
        self.q=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.k=nn.Linear(hidden_dim,num_kv_heads*self.head_dim,bias=False)
        self.v=nn.Linear(hidden_dim,num_kv_heads*self.head_dim,bias=False)
        self.o=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.reset_cache()
    def reset_cache(self):
        self.cache_k=self.cache_v=self.cache_positions=None
    def forward(self,x,position_ids=None,use_cache=False):
        b,s,d=x.shape
        offset=0 if self.cache_k is None or not use_cache else self.cache_k.shape[-2]
        if position_ids is None:position_ids=torch.arange(offset,offset+s,device=x.device)[None].expand(b,-1)
        q=self.q(x).view(b,s,self.num_q_heads,self.head_dim).transpose(1,2)
        k=self.k(x).view(b,s,self.num_kv_heads,self.head_dim).transpose(1,2)
        v=self.v(x).view(b,s,self.num_kv_heads,self.head_dim).transpose(1,2)
        q,k=apply_rope(q,position_ids),apply_rope(k,position_ids)
        key_positions=position_ids
        if use_cache and self.cache_k is not None:
            k=torch.cat((self.cache_k,k),dim=-2);v=torch.cat((self.cache_v,v),dim=-2)
            key_positions=torch.cat((self.cache_positions,position_ids),dim=-1)
        if use_cache:
            self.cache_k,self.cache_v,self.cache_positions=k.detach(),v.detach(),key_positions.detach()
        groups=self.num_q_heads//self.num_kv_heads
        k=k.repeat_interleave(groups,dim=1);v=v.repeat_interleave(groups,dim=1)
        mask=(position_ids[:,:,None]>=key_positions[:,None,:]).unsqueeze(1)
        result=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        return self.o(result.transpose(1,2).reshape(b,s,d))


def main():
    torch.manual_seed(42)
    model=GroupedQueryAttention(32,4,2).eval()
    x=torch.randn(1,7,32)
    with torch.no_grad():
        reference=model(x)
        a=model(x[:,:5],use_cache=True)
        b=model(x[:,5:],use_cache=True)
    torch.testing.assert_close(reference,torch.cat((a,b),dim=1),atol=1e-6,rtol=1e-5)
    print("GQA prefill + cached decode equals full causal attention")

if __name__=="__main__":main()
