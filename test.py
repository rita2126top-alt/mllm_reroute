import torch
import numpy as np
import torch.nn as nn
def applyrope(x,pos_ids):
    bsz,nhead,sqlen,headdim=x.shape
    theta=10000**(-2*torch.arrange(0,headdim//2,device=x.edvice)/headdim)
    freqs=pos_ids[..., None]*theta[None.None,:]
    cos=torch.cos(freqs)
    sin=torch.sin(freqs)
    cos=cos[:,None,:,:]
    sin=sin[:,None,:,:]
    x1=x[...,0::2]
    x2=x[...,1::2]
    x_rot=torch.cat([x1*cos-x2*sin,x1*sin+x2*cos],dim=1)
    return x_rot
class GQA(nn.Module):
    def _init_(self,hidden_dim,num_q_heads,num_kvgroups):
        super()._init_():
        self.hidden_dim=hidden_dim
        self.num_q_heads=num_q_heads
        self.num_kvgroups=num_kvgroups
        self.qpergroup=num_q_heads//num_kvgroups
        self.headdim=hidden_dim//num_q_heads
        self.wq=nn.Linear(hidden_dim,num_q_heads*self.headdim,bias=False)
        self.wk=nn.Linear(hidden_dim,num_kvgroups*self.headdim,bias=False)
        self.wv=nn.Linear(hidden_dim,num_kvgroups*self.headdim,bias=False)
        self.wo=nn.Linear(num_q_heads*self.headdim,hidden_dim,bias=False)
        self.cache_k=None
        self.cache_v=None
        def reset_cache(self):
            self.cache_k=None
            self.cache_v=None
        def splitheadsq(self,x):
            bsz,seq,_=x.shape
            return x.view(bsz,seq,self.num_q_heads,self.headdim).permute(0,2,1,3)
        def splitheadskv(self,x):
            bsz,seq,_=x.shape
            return x.view(bsz,seq,self.num_kvgroups,self.headdim).permute(0,2,1,3)
        def forward(self,x,pos_ids,isprefill):
            bsz,curseqlen,_=x.shape
            q=self.wq(x)
            k=self.wk(x)
            v=self.wv(x)
            q=splitheadsq(q)
            k=splitheadskv(k)
            v=splitheadskv(v)