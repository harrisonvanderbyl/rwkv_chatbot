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
             w[f"blocks.{x}.att.time_mix_k"],
             w[f"blocks.{x}.att.time_mix_v"],
             w[f"blocks.{x}.att.time_mix_r"],
             w[f"blocks.{x}.att.key.weight"],
             w[f"blocks.{x}.att.value.weight"],
             w[f"blocks.{x}.att.receptance.weight"],
             w[f"blocks.{x}.att.output.weight"],
             w[f"blocks.{x}.ffn.time_mix_k"],
             w[f"blocks.{x}.ffn.time_mix_r"],
             w[f"blocks.{x}.ffn.key.weight"],
             w[f"blocks.{x}.ffn.receptance.weight"],
             w[f"blocks.{x}.ffn.value.weight"],
         ], range(n_layer), []),
         [w["ln_out.weight"], w["ln_out.bias"],
          w["head.weight"]
          ]]
    torch.save(w, model_name+"_converted.pth")


class RWKV_PREPROCESS(nn.Module):
    def __init__(self, preProcess):
        super().__init__()
        self.preProcess = torch.Tensor(preProcess)

    def forward(self, x: torch.LongTensor):
        z = self.preProcess[x[0]]/len(x)
        for r in x[1:]:
            z += self.preProcess[r]/len(x)
        return z


class RWKV_POSTPROCESS(nn.Module):
    def __init__(self, postprocess):
        super().__init__()

        self.postProcess = postprocess

    def forward(self, x: torch.Tensor):
        return (self.postProcess[2] @ torch.layer_norm(
                x, (self.postProcess[0].shape[0],), weight=self.postProcess[0], bias=self.postProcess[1]))


class RWKV_LAYER(nn.Module):
    def __init__(self, w, offset):
        super().__init__()
        self.offset = offset
        self.ln1w = w[0::18]
        self.ln1b = w[1::18]
        self.ln2w = w[2::18]
        self.ln2b = w[3::18]
        self.time_decay = w[4::18]
        self.time_first = w[5::18]
        self.time_mix_k = w[6::18]
        self.time_mix_v = w[7::18]
        self.time_mix_r = w[8::18]
        self.key = w[9::18]
        self.value = w[10::18]
        self.receptance = w[11::18]
        self.outputv = w[12::18]
        self.time_mix_k_ffn = w[13::18]
        self.time_mix_r_ffn = w[14::18]
        self.key_ffn = w[15::18]
        self.receptance_ffn = w[16::18]
        self.value_ffn = w[17::18]

        print(len(self.outputv), len(self.ln1w), offset)

        self.n_layer = len(self.ln1w)
        self.eval()
        gc.collect()
        torch.cuda.empty_cache()

    def FF(self, sx, ln2w, ln2b, statex, i: int, time_mix_k, time_mix_r, kw, vw, rw):
        state = statex
        x = torch.layer_norm(sx, (ln2w.shape[0],), weight=ln2w, bias=ln2b)
        xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
        state[5*i+0] = x

        r = torch.sigmoid((rw @ xr))
        dx = (kw @ xk)
        clamped = torch.relu(dx)
        k = torch.square(clamped)
        kv = (vw @ k)
        return sx+(r * kv), state

    def SA(self, sx: torch.Tensor, ln1w, ln1b, state, i: int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw: torch.Tensor, vw, rw, ow):

        x = torch.layer_norm(
            sx, (ln1w.shape[0],), weight=ln1w, bias=ln1b)

        xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)

        state[5*i+1] = x

        r = torch.sigmoid((rw @ xr))
        k = (kw @ xk)
        v = (vw @ xv)

        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)

        a = e1 * aa + e2 * v
        b = e1 * bb + e2

        ww = pp + time_decay
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p

        rwkv = (r * a) / b
        return sx+(ow @ rwkv), state

    def forward(self, x: torch.Tensor, state: torch.Tensor):

        with torch.no_grad():
            for i in range(len(self.ln1w)):

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

                rx, state = self.FF(sx, ln2w, ln2b, state, i+self.offset,
                                    tmk, tmr, tmkw, tmvw, tmrw)

                x = rx

            return x, state


def empty_state(n_emb, layers, floatMode, device):
    state = torch.zeros(
        layers * 5, n_emb, device=device, dtype=floatMode)
    for i in range(layers):
        state[5*i+4] -= 1e30
    return state


def createRWKVModules(Path, RunDevice, FloatMode, chunkSize):

    def setToProp(x):
        x = x.to(dtype=FloatMode, device=RunDevice)
        return x

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
        setToCpu(w[0]))

    PostProcess = RWKV_POSTPROCESS(
        list(map(setToCpu, w[2])))
    Layers: list(RWKV_LAYER) = []
    print(len(w[1]))
    groups = chunkSize
    for i in range(len(w[1]))[::18*groups]:
        print(i)
        mm = w[1][i:i+18*groups]
        print(len(mm), "mm")
        Layers: List[RWKV_LAYER] = Layers+[RWKV_LAYER(
            list(map(setToProp, mm)), int(i/18))]

    return PreProcess, Layers, PostProcess
