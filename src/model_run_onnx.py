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


class RWKV_RNN(nn.Module):
    def __init__(self, args, argsnumns):
        super().__init__()

        self.args = args
        self.argsnumns = argsnumns
        self.FLOAT_MODE = torch.float32 if args["FLOAT_MODE"] == "fp32" else torch.float16 if args[
            "FLOAT_MODE"] == "fp16" else torch.bfloat16
        self.RUN_DEVICE = args["RUN_DEVICE"]
        self.n_layer = 0

        with torch.no_grad():
            w: Dict[str, torch.Tensor] = torch.load(
                args["MODEL_NAME"], map_location='cpu')
            self.n_emb = len(w['blocks.0.ln1.weight'])
            # refine weights and send to correct device
            keys = list(w.keys())
            if 'pos_emb_x' in keys:
                w['pos_emb'] = (w['pos_emb_x'] + w['pos_emb_y']
                                ).reshape(argsnumns["ctx_len"]+1, -1)[:-1, :]
            keys = list(w.keys())
            print_need_newline = False
            for x in keys:
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                    if DEBUG_TIME:
                        print(x, w[x].numpy())
                if '.time_decay' in x:
                    w[x] = w[x].float()
                    w[x] = -torch.exp(w[x])

                w[x] = w[x].to(dtype=self.FLOAT_MODE, device=self.RUN_DEVICE)

                w[x].requires_grad = False
                try:
                    if (int(x.split('.')[1])+1 > self.n_layer):
                        self.n_layer = int(x.split('.')[1])+1
                except:
                    pass

                if ('blocks.' not in x) or ('blocks.0.' in x):
                    if print_need_newline:
                        print('\n', end='')
                        print_need_newline = False
                    print(x.ljust(40), str(w[x].dtype).replace(
                        'torch.', '').ljust(10), w[x].device)

                else:
                    print_need_newline = True
                    print(
                        '.' if "cpu" in f'{w[x].device}' else "x", end='', flush=True)

        # store weights in self.w

        keys = list(w.keys())

        self.w = [w["emb.weight"], w["blocks.0.ln0.weight"],
                  w["blocks.0.ln0.bias"]] +\
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
            ], range(self.n_layer), []) +\
            [w["ln_out.weight"], w["ln_out.bias"],
             w["head.weight"]
             ]
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

            w = self.w

            x: torch.Tensor = w[0][int(ctx[0])]

            x = torch.layer_norm(
                x, (self.n_emb,), weight=self.w[1], bias=self.w[2])
            for o in range(self.n_layer):
                i = o

                startpos = i*18 + 3

                d = w[startpos:startpos+18]

                ln1w = d[0]
                ln1b = d[1]

                ln2w = d[2]
                ln2b = d[3]

                atc = d[4]
                atf = d[5]

                atmk = d[6]
                atmv = d[7]
                atmr = d[8]

                atd = d[9]
                avw = d[10]

                arw = d[11]
                aow = d[12]

                tmk = d[13]
                tmr = d[14]

                tmkw = d[15]
                tmrw = d[16]
                tmvw = d[17]

                sx, state = self.SA(x, ln1w, ln1b, state, i,
                                    atmk, atmv, atmr, atf, atc, atd, avw, arw, aow
                                    )

                rx, state = self.FF(sx, ln2w, ln2b, state, i,
                                    tmk, tmr, tmkw, tmvw, tmrw)

                x = rx

            return (w[-1] @ torch.layer_norm(
                x, (self.n_emb,), weight=w[-3], bias=w[-2])), state

    @ torch.jit.export
    def empty_state(self):
        device = self.RUN_DEVICE
        state = torch.zeros(
            self.n_layer * 5, self.n_emb, device=device, dtype=self.FLOAT_MODE)
        for i in range(self.n_layer):
            state[5*i+4] -= 1e30
        return state
