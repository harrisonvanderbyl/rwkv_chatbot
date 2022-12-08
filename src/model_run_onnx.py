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
        rm, = xx[self.m]

        out = self.preProcess[rm]
        return out, state


class RWKV_POSTPROCESS(nn.Module):
    def __init__(self, postprocess, device):
        super().__init__()

        self.postProcess0 = postprocess[0].to(device=device)
        self.postProcess1 = postprocess[1].to(device=device)
        self.postProcess2 = postprocess[2].to(device=device)
        self.m = torch.Tensor([0]).to(dtype=torch.int64)

    def forward(self, x: torch.Tensor, state):

        zz = torch.layer_norm(
            x, self.postProcess0.shape, weight=self.postProcess0, bias=self.postProcess1)
        out = torch.einsum('ik,k->i', [self.postProcess2, zz])
        return out, state


class RWKV_LAYER(nn.Module):
    def __init__(self, w, offset, dtypein=torch.int64, isStreamed=False):
        super().__init__()
        self.stream = lambda x: x
        if (isStreamed):
            self.stream = lambda x: x.to(device='cuda', non_blocking=True)

        def ispin(x):
            print(x.device)
            if (isStreamed):
                print("pinning memory")
                return x  # .pin_memory()
            else:
                return x

        # w = torch.Tensor(list(arr)).to(
        #     dtype=dtype, device=device)

        self.offset = offset
        self.ln1w = ispin(torch.stack(w[0::18]))
        self.ln1b = ispin(torch.stack(w[1::18]))
        self.ln2w = ispin(torch.stack(w[2::18]))
        self.ln2b = ispin(torch.stack(w[3::18]))
        self.time_decay = ispin(torch.stack(w[4::18]))
        self.time_first = ispin(torch.stack(w[5::18]))

        tk = w[6::18]
        tv = w[7::18]
        tr = w[8::18]
        kk = w[9::18]
        vv = w[10::18]
        rr = w[11::18]
        mm = []
        mn = []
        for i in range(len(kk)):
            mm = mm + [torch.stack(
                [kk[i], vv[i], rr[i]])]
            mn = mn + [torch.stack(
                [kk[i]@tk[i].diag(), vv[i]@tv[i].diag(), rr[i]@tr[i].diag()])]

        self.key = ispin(torch.stack(mm))
        self.keymul = ispin(torch.stack(mn)).reshape(
            len(mn), 3, mn[0].shape[1], mn[0].shape[1])
        self.outputv = ispin(torch.stack(w[12::18]))
        self.time_mix_k_ffn = ispin(torch.stack(w[13::18]))
        self.time_mix_r_ffn = ispin(torch.stack(w[14::18]))
        self.key_ffn = ispin(torch.stack(w[15::18]))
        self.receptance_ffn = ispin(torch.stack(w[16::18]))
        self.value_ffn = ispin(torch.stack(w[17::18]))
        print(len(self.outputv), len(self.ln1w), offset)

        self.n_layer = len(self.ln1w)
        self.m = torch.LongTensor([0]).to(dtype=dtypein)
        self.f = torch.LongTensor([5]).to(dtype=dtypein)

        print(self.m.dtype)
        self.layerlist = list(
            map(lambda x: x, list(range(self.n_layer))))
        self.cint = (self.offset+len(self.layerlist))*5
        self.uncint = (self.offset*7)
        print(self.layerlist)
        self.eval()
        gc.collect()
        torch.cuda.empty_cache()

    def FF(self, sx: torch.Tensor, ln2w, ln2b, time_mix_k: torch.Tensor, time_mix_r: torch.Tensor, kw: torch.Tensor, vw: torch.Tensor, rw: torch.Tensor, s0):

        x = torch.layer_norm(sx, (ln2w.shape[0],), weight=ln2w, bias=ln2b)
        dx = torch.addcmul(x, s0, time_mix_k)
        kwdx = torch.einsum('ik,k->i', [kw, dx])
        xr = torch.addcmul(x, s0, time_mix_r)
        rwxr = torch.einsum('ik,k->i', [rw, xr])
        r = torch.sigmoid(rwxr)
        clamped = torch.relu(kwdx)
        k = torch.square(clamped)
        kv = torch.einsum('ik,k->i', [vw, k])
        rkv = torch.mul(r, kv)
        output = torch.add(sx, rkv)

        return output, x

    def SA(self, sx: torch.Tensor, ln1w, ln1b, time_first: torch.Tensor, time_decay: torch.Tensor, kw: torch.Tensor, ow: torch.Tensor, s1, s2, s3, s4):

        x = torch.layer_norm(
            sx, (ln1w.shape[0],), weight=ln1w, bias=ln1b)

        rrk = (kw @ x)
        # print(rrk.shape, rkmul.shape, s1.shape)
        rrk2 = rrk+s1

        k = rrk2[0]
        v = rrk2[1]
        rr = rrk2[2]

        rsig = torch.sigmoid(rr)
        r = torch.mul(ow, rsig)
        aa = s2
        bb = s3
        pp = s4
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

        return output, x, torch.add(e1aa, e2v), torch.add(e1bb, e2), p

    def forward(self, x, state: torch.Tensor):
        bef = state[:self.uncint]
        outbet = []

        for i in bef:
            outbet.append(i)

        aft = state[self.cint:]

        ln1w = self.stream(self.ln1w)
        ln1b = self.stream(self.ln1b)
        ln2w = self.stream(self.ln2w)
        ln2b = self.stream(self.ln2b)
        time_decay = self.stream(self.time_decay)
        time_first = self.stream(self.time_first)
        key = self.stream(self.key)
        keymul = self.stream(self.keymul)
        outputv = self.stream(self.outputv)
        time_mix_k_ffn = self.stream(self.time_mix_k_ffn)
        time_mix_r_ffn = self.stream(self.time_mix_r_ffn)
        key_ffn = self.stream(self.key_ffn)
        receptance_ffn = self.stream(self.receptance_ffn)
        value_ffn = self.stream(self.value_ffn)

        bstate = state[self.offset*5::5]

        # print(bstate.shape)
        # print(keymul.shape)
        # matmul each layer on bstate and keymul, shape = (12,3,768,768), (12,768) -> (12,3,768)
        bstate = torch.einsum('ijkv,iv->ijk', [keymul, bstate])

        # print(bstate.shape)

        with torch.no_grad():
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

                s0 = bstate[i]

                s3 = state[i*5+self.offset*5+1]
                s4 = state[i*5+self.offset*5+2]
                s5 = state[i*5+self.offset*5+3]
                s6 = state[i*5+self.offset*5+4]

                sx, o0, o3, o4, o5 = self.SA(x, ln1wa, ln1ba, atf, atc, atd, aow, s0, s3, s4, s5
                                             )

                x, o6 = self.FF(sx, ln2wa, ln2ba,
                                tmk, tmr, tmkw, tmvw, tmrw, s6)
                outbet.append(o0)
                outbet.append(o3)
                outbet.append(o4)
                outbet.append(o5)
                outbet.append(o6)
            for i in aft:
                outbet.append(i)

            return x, torch.cat(outbet).reshape([len(outbet), outbet[0].shape[0]])


def empty_state(n_emb, layers, floatMode, device):
    state = torch.zeros(layers * 5,
                        n_emb, device=device[0] if device[0] == "cpu" else "cuda", dtype=floatMode)
    # for i in range(layers):
    #     state[5*i+4] -= 1e30
    # state = (*state,)
    return state


def createRWKVModules(Path, RunDevice, FloatMode, chunkSize, inttype=torch.int64):

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
        list(map(setToProp(0), w[2])), "cpu" if "cpu" in RunDevice[0] else "cuda")
    Layers: list(RWKV_LAYER) = []
    print(len(w[1]))
    groups = chunkSize
    for i in range(len(w[1]))[::18*groups]:
        print(i)
        mm = w[1][i:i+18*groups]
        print(len(mm), "mm")
        modelLayer = RWKV_LAYER(
            list(map(setToProp(int(i/(18))), mm)), int(i/18), inttype, "cuda" not in RunDevice[int(i/(18))] and "cpu" not in RunDevice[int(i/(18))])
        # modelLayer = torch.jit.script(
        #     modelLayer, (PreProcess.forward([127])))

        # modelLayer = torch.jit.optimize_for_inference(modelLayer)
        # torch.jit.enable_onednn_fusion(modelLayer)
        Layers: List[RWKV_LAYER] = Layers+[modelLayer]

    return PreProcess, Layers, PostProcess, int(len(w[1])/18)
