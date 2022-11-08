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

    w = [[w["emb.weight"], w["blocks.0.ln0.weight"],
          w["blocks.0.ln0.bias"]],
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


class RWKV_RNN(nn.Module):
    def __init__(self, args, argsnumns):
        super().__init__()

        self.args = args
        self.argsnumns = argsnumns
        self.FLOAT_MODE = torch.float32 if args["FLOAT_MODE"] == "fp32" else torch.float16 if args[
            "FLOAT_MODE"] == "fp16" else torch.bfloat16
        self.RUN_DEVICE = args["RUN_DEVICE"]

        if (not "_converted" in args["MODEL_NAME"]):
            createTensors(args["MODEL_NAME"][:-4])
            w: List(List(torch.Tensor)) = torch.load(
                args["MODEL_NAME"][:-4]+"_converted.pth", map_location=self.RUN_DEVICE)
        else:
            w: List(List(torch.Tensor)) = torch.load(
                args["MODEL_NAME"], map_location=self.RUN_DEVICE)

        self.preProcess = w[0]
        self.ln1w = w[1][0::18]
        self.ln1b = w[1][1::18]
        self.ln2w = w[1][2::18]
        self.ln2b = w[1][3::18]
        self.time_decay = w[1][4::18]
        self.time_first = w[1][5::18]
        self.time_mix_k = w[1][6::18]
        self.time_mix_v = w[1][7::18]
        self.time_mix_r = w[1][8::18]
        self.key = w[1][9::18]
        self.value = w[1][10::18]
        self.receptance = w[1][11::18]
        self.outputv = w[1][12::18]
        self.time_mix_k_ffn = w[1][13::18]
        self.time_mix_r_ffn = w[1][14::18]
        self.key_ffn = w[1][15::18]
        self.receptance_ffn = w[1][16::18]
        self.value_ffn = w[1][17::18]
        self.postProcess = w[2]

        def setToProp(x):
            x = x.to(dtype=self.FLOAT_MODE, device=self.RUN_DEVICE)
            return x

        def setToCpu(x):
            x = x.to(dtype=self.FLOAT_MODE, device="cpu")
            return x

        self.preProcess = list(map(setToProp, self.preProcess))
        self.ln1w = list(map(setToProp, self.ln1w))
        self.ln1b = list(map(setToProp, self.ln1b))
        self.ln2w = list(map(setToProp, self.ln2w))
        self.ln2b = list(map(setToProp, self.ln2b))
        self.time_decay = list(map(setToProp, self.time_decay))
        self.time_first = list(map(setToProp, self.time_first))
        self.time_mix_k = list(map(setToProp, self.time_mix_k))
        self.time_mix_v = list(map(setToProp, self.time_mix_v))
        self.time_mix_r = list(map(setToProp, self.time_mix_r))
        self.key = list(map(setToProp, self.key))
        self.value = list(map(setToProp, self.value))
        self.receptance = list(map(setToProp, self.receptance))
        self.outputv = list(map(setToProp, self.outputv))
        self.time_mix_k_ffn = list(map(setToProp, self.time_mix_k_ffn))
        self.time_mix_r_ffn = list(map(setToProp, self.time_mix_r_ffn))
        self.key_ffn = list(map(setToProp, self.key_ffn))
        self.receptance_ffn = list(map(setToProp, self.receptance_ffn))
        self.value_ffn = list(map(setToProp, self.value_ffn))
        self.postProcess = list(map(setToProp, self.postProcess))

        print(len(self.outputv), len(self.ln1w))

        self.n_layer = len(self.ln1w)
        self.n_emb = self.preProcess[1].shape[0]
        self.state = self.empty_state()
        self.eval()
        self.myEmptyState = self.empty_state()
        gc.collect()
        torch.cuda.empty_cache()

    def FF(self, sx, ln2w, ln2b, statex, i: int, time_mix_k, time_mix_r, kw, vw, rw):
        state = statex
        x = torch.layer_norm(sx, (self.n_emb,), weight=ln2w, bias=ln2b)
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
            sx, (self.n_emb,), weight=ln1w, bias=ln1b)

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

    def forward(self, ctx: torch.Tensor, state: torch.Tensor):

        with torch.no_grad():

            x = torch.layer_norm(
                ctx, (self.n_emb,), weight=self.preProcess[1], bias=self.preProcess[2])

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

                sx, state = self.SA(x, ln1w, ln1b, state, i,
                                    atmk, atmv, atmr, atf, atc, atd, avw, arw, aow
                                    )

                rx, state = self.FF(sx, ln2w, ln2b, state, i,
                                    tmk, tmr, tmkw, tmvw, tmrw)

                x = rx

            return (self.postProcess[2] @ torch.layer_norm(
                x, (self.n_emb,), weight=self.postProcess[0], bias=self.postProcess[1])), state

    @ torch.jit.export
    def empty_state(self):
        device = self.RUN_DEVICE
        state = torch.zeros(
            self.n_layer * 5, self.n_emb, device=device, dtype=self.FLOAT_MODE)
        for i in range(self.n_layer):
            state[5*i+4] -= 1e30
        return state
