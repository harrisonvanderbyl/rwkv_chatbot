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
from torch.nn import functional as F
import torch.nn as nn
from typing import List, Dict
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


@torch.jit.ignore
def sample(probs, temperature: float = 1.0, top_p_usual: float = 0.8) -> int:

    sorted_probs = torch.sort(probs, descending=True)[0]
    cumulative_probs = torch.cumsum(
        sorted_probs.float(), dim=-1).cpu().numpy()
    cutoff = float(sorted_probs[np.argmax(
        cumulative_probs > top_p_usual)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)

    out = torch.multinomial(probs.float(), 1, True)
    # print(sorted_probs[:3])
    return out


def isIn(a, b):
    for i in a:
        if i["ctx"] == b["ctx"]:
            return True
    return False


class RWKV_RNN(nn.Module):
    def __init__(self, args, argsnumns, layers: list[str]):
        super().__init__()

        self.args = args
        self.argsnumns = argsnumns
        self.FLOAT_MODE = args["FLOAT_MODE"]
        self.RUN_DEVICE = args["RUN_DEVICE"]
        self.layerdist = layers
        self.n_layer = 0
        self.procDevice = list(filter(
            lambda x: x != "proc", self.layerdist)).pop()
        if (self.procDevice == "proc"):
            self.procDevice = self.layerdist[-1]

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

                if self.FLOAT_MODE == "fp32":
                    w[x] = w[x].float()

                elif self.FLOAT_MODE == "bf16":
                    w[x] = w[x].bfloat16()

                elif self.FLOAT_MODE == "fp16":
                    if ('weight' in x or 'bias' in x) and 'ln' in x:
                        w[x] = w[x].half()
                    else:
                        w[x] = w[x].half()

                w[x].requires_grad = False
                try:
                    if (int(x.split('.')[1])+1 > self.n_layer):
                        self.n_layer = int(x.split('.')[1])+1
                except:
                    pass
                if args["RUN_DEVICE"] == "cuda" and x != 'emb.weight':

                    piece = x.split('.')[1]
                    if (piece in ["weight", "bias"]):
                        processedLayer = self.layerdist[self.n_layer-1]
                        if (processedLayer == "proc"):
                            processedLayer = self.procDevice
                    else:
                        processedLayer = self.layerdist[int(x.split('.')[1])]
                        if (processedLayer == "proc"):
                            processedLayer = "cpu"

                    w[x] = w[x].to(device=processedLayer, non_blocking=True)
                if (args["RUN_DEVICE"] == 'cuda'):
                    if (w[x].device.type == "cpu"):
                        w[x] = w[x].pin_memory()
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
        self.w = w

        self.eval()
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

    # @MyFunction

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

    def forward(self, ctx: list[int], state: torch.Tensor, preprocess_only: bool = False):
        with torch.no_grad():
            if (self.layerdist[0] != self.layerdist[self.n_layer-1]):
                state = state.to(device=self.layerdist[0])

            w = self.w

            x: torch.Tensor = (w["emb.weight"][ctx[-1]] *
                               4 + w["emb.weight"][ctx[-2]])*0.2

            if ("pos_emb" in w.keys()):
                pos_emb = w["pos_emb"][(len(ctx) % 1024)-1]
                x = x + pos_emb

            if self.RUN_DEVICE == 'cuda':
                x = x.to(device="cuda:0", non_blocking=True)

            for o in range(self.n_layer):
                i = o

                d: dict[str, torch.Tensor] = w

                d = {}
                for rr in w.keys():
                    if ("blocks."+str(i)+"." in rr):
                        if (self.RUN_DEVICE == "cuda" and self.layerdist[i] == "proc"):
                            d[rr] = w[rr].to(
                                self.procDevice, non_blocking=True)
                        else:
                            d[rr] = w[rr]

                ln1w = d["blocks."+str(i)+".ln1.weight"]
                ln1b = d["blocks."+str(i)+".ln1.bias"]
                tmk = d["blocks."+str(i)+".ffn.time_mix_k"]
                tmr = d["blocks."+str(i)+".ffn.time_mix_r"]
                tmkw = d["blocks."+str(i)+".ffn.key.weight"]
                tmvw = d["blocks."+str(i)+".ffn.value.weight"]
                tmrw = d["blocks."+str(i)+".ffn.receptance.weight"]
                ln2w = d["blocks."+str(i)+".ln2.weight"]
                ln2b = d["blocks."+str(i)+".ln2.bias"]
                atmk = d["blocks."+str(i)+".att.time_mix_k"]
                atmv = d["blocks."+str(i)+".att.time_mix_v"]
                atmr = d["blocks."+str(i)+".att.time_mix_r"]
                atf = d["blocks."+str(i)+".att.time_first"]
                atc = d["blocks."+str(i)+".att.time_decay"]
                atd = d["blocks."+str(i)+".att.key.weight"]
                avw = d["blocks."+str(i)+".att.value.weight"]
                arw = d["blocks."+str(i)+".att.receptance.weight"]
                aow = d["blocks."+str(i)+".att.output.weight"]

                if (i == 0):
                    x = torch.layer_norm(
                        x, (self.n_emb,), weight=d["blocks.0.ln0.weight"], bias=d["blocks.0.ln0.bias"])
                else:
                    if (self.layerdist[i] != self.layerdist[i-1]):
                        if (self.layerdist[i] == "proc"):
                            x = x.to(device=self.procDevice)
                            state = state.to(device=self.procDevice)
                        else:
                            x = x.to(device=self.layerdist[i])
                            state = state.to(device=self.layerdist[i])

                sx, state = self.SA(x, ln1w, ln1b, state, i,
                                    atmk, atmv, atmr, atf, atc, atd, avw, arw, aow
                                    )

                rx, state = self.FF(sx, ln2w, ln2b, state, i,
                                    tmk, tmr, tmkw, tmvw, tmrw)

                x = rx

                # Delete Procedurally Moved Layer
                if ((self.layerdist[i] == "proc")):

                    for rr in w.keys():
                        if ("blocks."+str(i)+"." in rr):

                            del d[rr]

            if preprocess_only:
                return x, state

            return (w["head.weight"] @ torch.layer_norm(
                x, (self.n_emb,), weight=w["ln_out.weight"], bias=w["ln_out.bias"])), state

    @ torch.jit.ignore
    def empty_state(self):
        device = self.layerdist[0]
        if (device == "proc"):
            if (self.RUN_DEVICE == "cuda"):
                device = self.procDevice
            else:
                device = "cpu"

        state = torch.zeros(
            self.n_layer * 5, self.n_emb, device=device, dtype=torch.float32 if self.FLOAT_MODE == "fp32" else torch.bfloat16 if self.FLOAT_MODE == "bf16" else torch.float16)
        for i in range(self.n_layer):
            state[5*i+4] -= 1e30
        return state

    @ torch.jit.ignore
    def loadContext(self, ctx: list[int] = [], newctx: list[int] = [187], statex=None, silent=False):

        if statex is None:
            statex = self.empty_state()

        m = tqdm if not silent else lambda x: x
        for i in m(range(len(newctx)-1)):
            x = ctx+newctx[:i+1]
            o, statex = self.forward(
                x, statex, preprocess_only=True)

        return ctx+newctx, statex

    @ torch.jit.ignore
    def sample_logits(self, ozut: torch.Tensor, x: List[int], temperature: float = 1.0, top_p_usual: float = 0.8):
        # out[self.UNKNOWN_CHAR] = -float('Inf')
       # out[self.UNKNOWN_CHAR] = -float('Inf')
        # turn to float if is half and cpu
        out = ozut
        if out.dtype == torch.half and out.device == torch.device('cpu'):
            out = out.float()
        probs = F.softmax(out, dim=-1)

        return sample(probs, temperature, top_p_usual)

    @ torch.jit.ignore
    def run(self, currstate: list({"score": float, "ctx": list[int], "state": torch.Tensor}), temp: float = 1.5, top_p: float = 0.9, nla: float = 0, endChars=[[187, 187], [535]]):
        options = []
        for i in range(len(currstate)):

            ctx = currstate[i]["ctx"]
            if any(list(map(lambda x: x == ctx[-len(x):], endChars))):
                currstate[i]["score"] *= 1.1
                options.append(currstate[i])
                continue

            state = currstate[i]["state"]
            score = currstate[i]["score"]

            out1, state = self.forward(ctx, state.clone())

            ttt = self.sample_logits(
                out1,
                ctx,
                temperature=0.8,
                top_p_usual=0.9,
            )

            for j in range(len(ttt)):
                options.append(
                    {"score": (score*-0.5+out1[ttt[j]]/100.0), "ctx": ctx+[ttt[j]], "state": state})

        options.sort(key=lambda x: x["score"], reverse=True)
        # remove duplicates using reduce

        options = reduce(lambda x, y: x if isIn(x, y) else x+[y], options, [])
        options = options[:4]
        # scores = list(map(lambda x: x["score"], options))
        # cumscore = sum(scores)
        # options = list(map(lambda x: {
        #                "score": x["score"]/cumscore, "ctx": x["ctx"], "state": x["state"]}, options))

        return options
