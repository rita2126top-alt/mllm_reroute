"""CPU-only mathematical decoder fixture; not a pretrained model test."""
from types import SimpleNamespace
import torch
from torch import nn
from models.patching import _apply_rotary_pos_emb

class Cache:
    def __init__(self, layers):
        self.keys, self.values = [None]*layers, [None]*layers
    def get_seq_length(self):
        return 0 if self.keys[0] is None else self.keys[0].shape[-2]

class Attention(nn.Module):
    def __init__(self, dim=16, heads=2):
        super().__init__()
        self.head_dim = dim//heads
        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x, pe, mask, cache, layer, use_cache):
        b,s,d=x.shape
        q,k,v=[f(x).view(b,s,-1,self.head_dim).transpose(1,2) for f in (self.q_proj,self.k_proj,self.v_proj)]
        if pe is not None:
            q,k=_apply_rotary_pos_emb(q,k,*pe)
        if use_cache:
            if cache.keys[layer] is not None:
                k=torch.cat((cache.keys[layer],k),dim=-2)
                v=torch.cat((cache.values[layer],v),dim=-2)
            cache.keys[layer],cache.values[layer]=k,v
        y=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=mask is None and s>1)
        return self.o_proj(y.transpose(1,2).reshape(b,s,d))

class Layer(nn.Module):
    def __init__(self, i, dim=16, tuple_output=False):
        super().__init__()
        self.i,self.tuple_output=i,tuple_output
        self.input_layernorm=nn.LayerNorm(dim)
        self.self_attn=Attention(dim)
        self.post_attention_layernorm=nn.LayerNorm(dim)
        self.mlp=nn.Sequential(nn.Linear(dim,32),nn.SiLU(),nn.Linear(32,dim))
    def forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_values=None,use_cache=False,position_embeddings=None,**kwargs):
        y=hidden_states+self.self_attn(self.input_layernorm(hidden_states),position_embeddings,attention_mask,past_key_values,self.i,use_cache)
        y=y+self.mlp(self.post_attention_layernorm(y))
        return (y,) if self.tuple_output else y

class ToyModel(nn.Module):
    def __init__(self, layers=6, dim=16, tuple_output=False):
        super().__init__()
        self.embedding=nn.Embedding(128,dim)
        self.model=nn.Module()
        self.model.language_model=nn.Module()
        self.model.language_model.layers=nn.ModuleList([Layer(i,dim,tuple_output) for i in range(layers)])
        self.head=nn.Linear(dim,128,bias=False)
        self.config=SimpleNamespace(image_token_index=99,vision_config=SimpleNamespace(patch_size=1))
    def forward(self,input_ids,pixel_values=None,attention_mask=None,past_key_values=None,use_cache=False,**kwargs):
        x=self.embedding(input_ids)
        n=x.shape[1]
        offset=0 if past_key_values is None else past_key_values.get_seq_length()
        pos=torch.arange(offset,offset+n,device=x.device)[None]
        hd=self.model.language_model.layers[0].self_attn.head_dim
        inv=10000**(-torch.arange(0,hd,2,device=x.device).float()/hd)
        frequencies=pos[:,:,None]*inv
        pe=torch.cat((frequencies,frequencies),-1)
        pe=(pe.cos().to(x.dtype),pe.sin().to(x.dtype))
        cache=past_key_values or (Cache(len(self.model.language_model.layers)) if use_cache else None)
        for layer in self.model.language_model.layers:
            y=layer(x,position_ids=pos,position_embeddings=pe,use_cache=use_cache,past_key_values=cache)
            x=y[0] if isinstance(y,tuple) else y
        return SimpleNamespace(logits=self.head(x),last_hidden_state=x,past_key_values=cache)

def inputs():
    return dict(input_ids=torch.tensor([[1,99,99,99,99,2,3]]),pixel_values=torch.ones(1,3,2,2))
