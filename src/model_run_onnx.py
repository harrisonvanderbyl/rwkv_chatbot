########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from ast import Delete
from functools import reduce
import types
import torch
import math
import os
import gc
import torch.nn as nn
from typing import List, Dict, Tuple, Union
from torch import autocast
import numpy as np
from tqdm import tqdm
# Make sure to use nightly build of torchdynamo
# import torchdynamo
# MyFunction = torchdynamo.optimize(
#     "nvfuser")  # !!!BUGGY!!! wrong output

RWKV_HEAD_QK_DIM = 0
print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM}\n')

DEBUG_TIME = False   # True False - show trained time-coeffs


def createTensors(model_name):
    n_layer = 0

    with torch.no_grad():
        w: Dict[str, torch.Tensor] = torch.load(
            model_name+".pth", map_location='cpu')
        # refine weights and send to correct device
        keys = list(w.keys())
        for x in keys:
            if '.time_' in x:
                w[x] = w[x].squeeze()
                if DEBUG_TIME:
                    print(x, w[x].numpy())
            if '.time_decay' in x:
                w[x] = w[x].float()
                w[x] = -torch.exp(w[x])

            w[x].requires_grad = False
            w[x] = w[x].to(dtype=torch.bfloat16)
            try:
                if (int(x.split('.')[1])+1 > n_layer):
                    n_layer = int(x.split('.')[1])+1
            except:
                pass

    # store weights in self.w

    keys = list(w.keys())

    z = []
    for x in tqdm(range(len(w["emb.weight"]))):
        z = z + [torch.layer_norm(w["emb.weight"][x], (w["blocks.0.ln0.weight"].shape[0],),
                                  weight=w["blocks.0.ln0.weight"], bias=w["blocks.0.ln0.bias"]).float().numpy()]

    w = [torch.Tensor(z).to(dtype=torch.bfloat16),
         reduce(lambda y, x: y+[
             w[f"blocks.{x}.ln1.weight"],
             w[f"blocks.{x}.ln1.bias"],
             w[f"blocks.{x}.ln2.weight"],
             w[f"blocks.{x}.ln2.bias"],
             w[f"blocks.{x}.att.time_decay"],
             w[f"blocks.{x}.att.time_first"],
             (1 / w[f"blocks.{x}.att.time_mix_k"]-1),
             (1 / w[f"blocks.{x}.att.time_mix_v"]-1),
             (1 / w[f"blocks.{x}.att.time_mix_r"]-1),
             w[f"blocks.{x}.att.key.weight"]*w[f"blocks.{x}.att.time_mix_k"],
             w[f"blocks.{x}.att.value.weight"]*w[f"blocks.{x}.att.time_mix_v"],
             w[f"blocks.{x}.att.receptance.weight"] *
             w[f"blocks.{x}.att.time_mix_r"],
             w[f"blocks.{x}.att.output.weight"],
             (1 / w[f"blocks.{x}.ffn.time_mix_k"]-1),
             (1 / w[f"blocks.{x}.ffn.time_mix_r"]-1),
             w[f"blocks.{x}.ffn.key.weight"]*w[f"blocks.{x}.ffn.time_mix_k"],
             w[f"blocks.{x}.ffn.receptance.weight"] *
             w[f"blocks.{x}.ffn.time_mix_r"],
             w[f"blocks.{x}.ffn.value.weight"],
         ], range(n_layer), []),
         [w["ln_out.weight"], w["ln_out.bias"],
          w["head.weight"]
          ]]
    torch.save(w, model_name+"_converted.pth")


class RWKV_PREPROCESS(nn.Module):
    def __init__(self, preProcess):
        super().__init__()
        self.preProcess = preProcess

    def forward(self, x: torch.LongTensor):
        return self.preProcess[x[0]]


class RWKV_POSTPROCESS(nn.Module):
    def __init__(self, postprocess: torch.Tensor):
        super().__init__()

        self.postProcess0 = postprocess[0]
        self.postProcess1 = postprocess[1]
        self.postProcess2 = postprocess[2]

    def forward(self, x: torch.Tensor):
        zz = torch.layer_norm(
            x, self.postProcess0.shape, weight=self.postProcess0, bias=self.postProcess1)
        out = torch.einsum('ik,k->i', [self.postProcess2, zz])
        return out


class RWKV_LAYER(nn.Module):
    def __init__(self, w, offset):
        super().__init__()

        dtype = w[0].dtype
        device = w[0].device

        w = list(map(lambda x: x.float().cpu().numpy(), w))

        # w = torch.Tensor(list(arr)).to(
        #     dtype=dtype, device=device)

        self.offset = offset
        self.ln1w = torch.tensor(w[0::18]).to(
            dtype=dtype, device=device)
        self.ln1b = torch.tensor(w[1::18]).to(dtype=dtype, device=device)
        self.ln2w = torch.tensor(w[2::18]).to(dtype=dtype, device=device)
        self.ln2b = torch.tensor(w[3::18]).to(dtype=dtype, device=device)
        self.time_decay = torch.tensor(w[4::18]).to(dtype=dtype, device=device)
        self.time_first = torch.tensor(w[5::18]).to(dtype=dtype, device=device)
        self.time_mix_k = torch.tensor(w[6::18]).to(dtype=dtype, device=device)
        self.time_mix_v = torch.tensor(w[7::18]).to(dtype=dtype, device=device)
        self.time_mix_r = torch.tensor(w[8::18]).to(dtype=dtype, device=device)
        self.key = torch.tensor(w[9::18]).to(dtype=dtype, device=device)
        self.value = torch.tensor(w[10::18]).to(dtype=dtype, device=device)
        self.receptance = torch.tensor(
            w[11::18]).to(dtype=dtype, device=device)
        self.outputv = torch.tensor(w[12::18]).to(dtype=dtype, device=device)
        self.time_mix_k_ffn = torch.tensor(
            w[13::18]).to(dtype=dtype, device=device)
        self.time_mix_r_ffn = torch.tensor(
            w[14::18]).to(dtype=dtype, device=device)
        self.key_ffn = torch.tensor(w[15::18]).to(dtype=dtype, device=device)
        self.receptance_ffn = torch.tensor(
            w[16::18]).to(dtype=dtype, device=device)
        self.value_ffn = torch.tensor(w[17::18]).to(dtype=dtype, device=device)

        print(len(self.outputv), len(self.ln1w), offset)

        self.n_layer = len(self.ln1w)
        self.eval()
        gc.collect()
        torch.cuda.empty_cache()

    def FF(self, sx: torch.Tensor, ln2w, ln2b, statex, i: int, time_mix_k: torch.Tensor, time_mix_r: torch.Tensor, kw: torch.Tensor, vw: torch.Tensor, rw: torch.Tensor):
        state = statex
        x = torch.layer_norm(sx, (ln2w.shape[0],), weight=ln2w, bias=ln2b)
        dx = torch.addcmul(x, state[5*i+0], time_mix_k)
        kwdx = torch.einsum('ik,k->i', [kw, dx])
        xr = torch.addcmul(x, state[5*i+0], time_mix_r)
        rwxr = torch.einsum('ik,k->i', [rw, xr])
        r = torch.sigmoid(rwxr)
        clamped = torch.relu(kwdx)
        k = torch.square(clamped)
        kv = torch.einsum('ik,k->i', [vw, k])
        rkv = torch.mul(r, kv)
        output = torch.add(sx, rkv)
        state[5*i+0] = x
        return output, state

    def SA(self, sx: torch.Tensor, ln1w, ln1b, state: torch.Tensor, i: int, time_mix_k: torch.Tensor, time_mix_v: torch.Tensor, time_mix_r: torch.Tensor, time_first: torch.Tensor, time_decay: torch.Tensor, kw: torch.Tensor, vw: torch.Tensor, rw: torch.Tensor, ow: torch.Tensor):

        x = torch.layer_norm(
            sx, ln1w.shape, weight=ln1w, bias=ln1b)

        xtk = torch.addcmul(x, state[5*i+1], time_mix_k)

        k = torch.einsum('ik,k->i', [kw, xtk])

        vtk = torch.addcmul(x, state[5*i+1], time_mix_v)

        v = torch.einsum('ik,k->i', [vw, vtk])

        rtk = torch.addcmul(x, state[5*i+1], time_mix_r)

        rr = torch.einsum('ik,k->i', [rw, rtk])

        rsig = torch.sigmoid(rr)
        r = torch.mul(ow, rsig)
        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = torch.add(time_first, k)
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)

        e1aa = torch.mul(e1, aa)
        e2v = torch.mul(e2, v)
        e1bb = torch.mul(e1, bb)

        a = torch.add(e1aa, e2v)
        b = torch.add(e1bb, e2)

        ww = torch.add(pp, time_decay)
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        e1bb = torch.mul(e1, bb)
        e1aa = torch.mul(e1, aa)
        e2v = torch.mul(e2, v)

        ab = torch.div(a, b)
        rwkv = torch.einsum('ik,k->i', [r, ab])
        output = torch.add(sx, rwkv)

        state[5*i+1] = x
        state[5*i+2] = torch.add(e1aa, e2v)
        state[5*i+3] = torch.add(e1bb, e2)
        state[5*i+4] = p
        return output, state

    def forward(self, x: torch.Tensor, state: torch.Tensor):

        with torch.no_grad():
            for i in range(self.ln1w.shape[0]):

                ln1w = self.ln1w[i]
                ln1b = self.ln1b[i]

                ln2w = self.ln2w[i]
                ln2b = self.ln2b[i]

                atc = self.time_decay[i]
                atf = self.time_first[i]

                atmk = self.time_mix_k[i]
                atmv = self.time_mix_v[i]
                atmr = self.time_mix_r[i]

                atd = self.key[i]
                avw = self.value[i]

                arw = self.receptance[i]
                aow = self.outputv[i]

                tmk = self.time_mix_k_ffn[i]
                tmr = self.time_mix_r_ffn[i]

                tmkw = self.key_ffn[i]
                tmrw = self.receptance_ffn[i]
                tmvw = self.value_ffn[i]

                sx, state = self.SA(x, ln1w, ln1b, state, i+self.offset,
                                    atmk, atmv, atmr, atf, atc, atd, avw, arw, aow
                                    )

                x, state = self.FF(sx, ln2w, ln2b, state, i+self.offset,
                                   tmk, tmr, tmkw, tmvw, tmrw)

            return x, state


def empty_state(n_emb, layers, floatMode, device):
    state = torch.zeros(
        layers * 5, n_emb, device=device[0], dtype=floatMode)
    for i in range(layers):
        state[5*i+4] -= 1e30
    return state


def createRWKVModules(Path, RunDevice, FloatMode, chunkSize):

    def setToProp(i):
        return lambda x: x.to(dtype=FloatMode, device=RunDevice[i])

    def setToCpu(x):
        x = x.to(dtype=FloatMode, device="cpu")
        return x

    if (not "_converted" in Path):
        createTensors(Path[: -4])
        w: List(List(torch.Tensor)) = torch.load(
            Path[:-4]+"_converted.pth", map_location="cpu")
    else:
        w: List(List(torch.Tensor)) = torch.load(
            Path, map_location="cpu")

    PreProcess = RWKV_PREPROCESS(
        setToProp(0)(w[0]))

    PostProcess = RWKV_POSTPROCESS(
        list(map(setToProp(-1), w[2])))
    Layers: list(RWKV_LAYER) = []
    print(len(w[1]))
    groups = chunkSize
    for i in range(len(w[1]))[::18*groups]:
        print(i)
        mm = w[1][i:i+18*groups]
        print(len(mm), "mm")
        modelLayer = RWKV_LAYER(
            list(map(setToProp(int(i/(18*groups))), mm)), int(i/18))
        # modelLayer = torch.jit.script(
        #     modelLayer, (PreProcess.forward([127])))

        # modelLayer = torch.jit.optimize_for_inference(modelLayer)
        # torch.jit.enable_onednn_fusion(modelLayer)
        Layers: List[RWKV_LAYER] = Layers+[modelLayer]

    return PreProcess, Layers, PostProcess, int(len(w[1])/18)
