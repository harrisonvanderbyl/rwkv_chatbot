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
    return w


class RWKV_PREPROCESS(nn.Module):
    def __init__(self, preProcess, device):
        super().__init__()
        self.preProcess = preProcess.to(device=device)
        self.m = torch.Tensor([0]).to(dtype=torch.int64)

    def forward(self, xx, state):

        out, = self.preProcess[xx]
        return out, state


class RWKV_POSTPROCESS(nn.Module):
    def __init__(self, postprocess, device, compatibility=False):
        super().__init__()
        self.mv = torch.mv
        if (compatibility):
            self.mv = lambda x, y: torch.einsum('ik,k->i', [x, y])

        self.postProcess0 = postprocess[0].to(device=device)
        self.postProcess1 = postprocess[1].to(device=device)
        self.postProcess2 = postprocess[2].to(device=device)
        self.m = torch.Tensor([0]).to(dtype=torch.int64)

    def forward(self, x: torch.Tensor, state):

        zz = torch.layer_norm(
            x.to(device=self.postProcess0.device), self.postProcess0.shape, weight=self.postProcess0, bias=self.postProcess1)
        out = self.mv(self.postProcess2, zz)
        return out, state


class RWKV_LAYER(nn.Module):
    def __init__(self, w, offset, dtypein=torch.int64, isStreamed=False, compatibility=False):
        super().__init__()
        self.stream = lambda x: x
        if (isStreamed):
            self.stream = lambda x: x.to(device='cuda', non_blocking=True)

        def ispin(x):
            print(x.device)
            if (isStreamed):
                print("pinning memory")
                return x.pin_memory()
            else:
                return x
        self.mv = torch.mv
        if (compatibility):
            self.mv = lambda x, y: torch.einsum('ik,k->i', [x, y])
        # w = torch.Tensor(list(arr)).to(
        #     dtype=dtype, device=device)

        self.offset = offset
        self.ln1w = ispin(torch.stack(w[0::18]))
        self.ln1b = ispin(torch.stack(w[1::18]))
        self.ln2w = ispin(torch.stack(w[2::18]))
        self.ln2b = ispin(torch.stack(w[3::18]))
        self.time_decay = ispin((torch.stack(w[4::18]).exp()))
        self.time_first = ispin((torch.stack(w[5::18]).exp()))

        tk = w[6::18]
        tv = w[7::18]
        tr = w[8::18]
        kk = w[9::18]
        vv = w[10::18]
        rr = w[11::18]
        mm = []
        for i in range(len(kk)):
            mm = mm + [torch.stack(
                [(kk[i]), vv[i], (-rr[i])])]

        self.vvtv = ispin(torch.stack(tv))
        self.kktk = ispin(torch.stack(tk))
        self.rrtr = ispin(torch.stack(tr))

        def r(x):
            print(x)
            self.outputonce = lambda x: x

        self.outputonce = r

        self.key = ispin(torch.stack(mm))
        self.outputv = ispin(torch.stack(w[12::18]))
        self.time_mix_k_ffn = ispin(torch.stack(w[13::18]))
        self.time_mix_r_ffn = ispin(torch.stack(w[14::18]))
        self.key_ffn = ispin(torch.stack(w[15::18]))
        self.receptance_ffn = ispin(-torch.stack(w[16::18]))
        self.value_ffn = ispin(torch.stack(w[17::18]))
        print(len(self.outputv), len(self.ln1w), offset)

        self.n_layer = len(self.ln1w)
        self.m = torch.LongTensor([0]).to(
            dtype=torch.int32 if compatibility else torch.int64)
        self.f = torch.LongTensor([5]).to(
            dtype=torch.int32 if compatibility else torch.int64)

        print(self.m.dtype)
        self.layerlist = list(
            map(lambda x: x, list(range(self.n_layer))))
        self.cint = (self.offset+len(self.layerlist))
        self.uncint = (self.offset)
        print(self.layerlist)
        self.eval()

        self.outputfunc = lambda state, statea, stateb, statec, stated: state
        if (compatibility):
            self.outputfunc = lambda state, statea, stateb, statec, stated: torch.stack(
                [statea, stateb, statec, stated])

        gc.collect()
        torch.cuda.empty_cache()

    def FF(self, sx: torch.Tensor, ln2w, ln2b, time_mix_k: torch.Tensor, time_mix_r: torch.Tensor, kw: torch.Tensor, vw: torch.Tensor, rw: torch.Tensor):

        x = torch.layer_norm(sx, (ln2w.shape[0],), weight=ln2w, bias=ln2b)

        k = self.mv(kw, x + time_mix_k).relu().square()

        r = self.mv(rw, x + time_mix_r).exp()

        kv = self.mv(vw, k)
        rkv = kv/(1+r)
        output = torch.add(sx, rkv)

        return output, x

    def SA(self, sx: torch.Tensor, ln1w, ln1b, time_first: torch.Tensor, time_decay: torch.Tensor, kw: torch.Tensor, ow: torch.Tensor, instateAK, instateAV, instateAR, instateB, instateC):

        x = torch.layer_norm(
            sx, (ln1w.shape[0],), weight=ln1w, bias=ln1b)

        k = self.mv(kw[0], x + instateAK).to(torch.float64).exp()

        v = self.mv(kw[1], x + instateAV)
        self.outputonce(v[0])

        r = self.mv(kw[2], x + instateAR).to(torch.float64).exp() + 1

        w = instateB + time_first*k*v
        d = instateC*r+time_first*k*r

        rwkv = self.mv(ow, (w/d).to(dtype=ow.dtype))
        output = sx+rwkv

        outstateA = x
        outstateB = instateB * time_decay + k * v  # ne33nd
        outstateC = instateC * time_decay + k

        return output, outstateA, outstateB, outstateC

    def forward(self, x, state: torch.Tensor):
        with torch.no_grad():

            ln1w = self.stream(self.ln1w)
            x = x.to(device=ln1w.device)
            state = state.to(device=x.device)
            ln1b = self.stream(self.ln1b)
            ln2w = self.stream(self.ln2w)
            ln2b = self.stream(self.ln2b)
            time_decay = self.stream(self.time_decay)
            time_first = self.stream(self.time_first)
            key = self.stream(self.key)
            outputv = self.stream(self.outputv)

            key_ffn = self.stream(self.key_ffn)
            receptance_ffn = self.stream(self.receptance_ffn)
            value_ffn = self.stream(self.value_ffn)

            viewer = state.view(state.shape)
            statea = viewer[0][self.uncint: self.cint]
            stateb = viewer[1][self.uncint: self.cint]
            statec = viewer[2][self.uncint: self.cint]
            stated = viewer[3][self.uncint: self.cint]

            vvtv = self.stream(self.vvtv)*statea.to(dtype=self.vvtv.dtype)
            kktk = self.stream(self.kktk)*statea.to(dtype=self.vvtv.dtype)
            rrtr = self.stream(self.rrtr)*statea.to(dtype=self.vvtv.dtype)

            time_mix_k_ffn = self.stream(
                self.time_mix_k_ffn)*stated.to(dtype=self.vvtv.dtype)
            time_mix_r_ffn = self.stream(
                self.time_mix_r_ffn)*stated.to(dtype=self.vvtv.dtype)

            for i in self.layerlist:

                ln1wa = ln1w[i]
                ln1ba = ln1b[i]

                ln2wa = ln2w[i]
                ln2ba = ln2b[i]

                atc = time_decay[i]
                atf = time_first[i]

                atd = key[i]

                aow = outputv[i]

                tmk = time_mix_k_ffn[i]
                tmr = time_mix_r_ffn[i]

                tmkw = key_ffn[i]
                tmrw = receptance_ffn[i]
                tmvw = value_ffn[i]

                # kmul = kmull[i]mul[1]@s1
                kmul0i = kktk[i]
                kmul1i = vvtv[i]
                kmul2i = rrtr[i]

                # kmul1 = kmul11[i]

                sx, statea[i], stateb[i], statec[i] = self.SA(x, ln1wa, ln1ba, atf, atc, atd, aow, kmul0i, kmul1i, kmul2i, stateb[i], statec[i]
                                                              )

                x, stated[i] = self.FF(sx, ln2wa, ln2ba,
                                       tmk, tmr, tmkw, tmvw, tmrw)

            return x, self.outputfunc(state, statea, stateb, statec, stated)


def empty_state(n_emb, layers, floatMode, device):
    state = torch.zeros(4, layers,
                        n_emb, device=device[0] if device[0] == "cpu" else "cuda", dtype=torch.float64)+0.01
    # for i in range(layers):
    #     state[5*i+4] -= 1e30
    # state = (*state,)
    return state


def createRWKVModules(Path, RunDevice, FloatMode, chunkSize, inttype=torch.int64, compat=False):

    def setToProp(i):
        def fx(x): return x

        cdev = RunDevice[i] if "cuda" in RunDevice[i] else "cpu"
        print(cdev, RunDevice[i])
        return lambda x: fx(x.to(dtype=FloatMode, device=cdev))

    def setToCpu(x):
        x = x.to(dtype=FloatMode, device="cpu")
        return x

    if (not "_converted" in Path):
        w = createTensors(Path[: -4])
    else:
        w: List(List(torch.Tensor)) = torch.load(
            Path, map_location="cpu")

    PreProcess = RWKV_PREPROCESS(
        setToProp(0)(w[0]), "cpu" if "cpu" in RunDevice[0] else "cuda")

    PostProcess = RWKV_POSTPROCESS(
        list(map(setToProp(0), w[2])), "cpu" if "cpu" in RunDevice[0] else "cuda", compatibility=compat)
    Layers: list(RWKV_LAYER) = []
    print(len(w[1]))
    groups = chunkSize
    for i in range(len(w[1]))[::18*groups]:
        print(i)
        mm = w[1][i:i+18*groups]
        print(len(mm), "mm")
        modelLayer = RWKV_LAYER(
            list(map(setToProp(int(i/(18))), mm)), int(i/18), inttype, "cuda" not in RunDevice[int(i/(18))] and "cpu" not in RunDevice[int(i/(18))], compatibility=compat)
        # modelLayer = torch.jit.script(
        #     modelLayer, (PreProcess.forward([127])))

        # modelLayer = torch.jit.optimize_for_inference(modelLayer)
        # torch.jit.enable_onednn_fusion(modelLayer)
        Layers: List[RWKV_LAYER] = Layers+[modelLayer]

    return PreProcess, Layers, PostProcess, int(len(w[1])/18)
