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
import src.tensorflowrwkv as tensorflowrwkv
import tensorflow as tf
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
    def __init__(self, preProcess, device, inttype=torch.int64):
        super().__init__()
        self.preProcess = preProcess.to(device=device)
        self.m = torch.Tensor([0]).to(dtype=inttype)

    def forward(self, xx, state):

        out, = self.preProcess[xx]
        return out, state


class RWKV_POSTPROCESS(nn.Module):
    def __init__(self, postprocess, device, compatibility=False, inttype=torch.int64):
        super().__init__()
        self.mv = torch.mv
        if (compatibility):
            self.mv = lambda x, y: torch.einsum('ik,k->i', [x, y])

        self.postProcess0 = postprocess[0].to(device=device)
        self.postProcess1 = postprocess[1].to(device=device)
        self.postProcess2 = postprocess[2].to(device=device)
        self.m = torch.Tensor([0]).to(dtype=inttype)

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
            # self.mv = lambda x, y: (x*y).sum(1)

            # w = torch.Tensor(list(arr)).to(
            #     dtype=dtype, device=device)

        self.offset = offset
        self.ln1w = ispin(torch.stack(w[0::18]))
        self.ln1b = ispin(torch.stack(w[1::18]))
        self.ln2w = ispin(torch.stack(w[2::18]))
        self.ln2b = ispin(torch.stack(w[3::18]))
        self.time_decay = ispin((torch.stack(w[4::18])))
        self.time_first = ispin((torch.stack(w[5::18])))

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
        print("keyffn", self.key_ffn.shape)
        self.receptance_ffn = ispin(-torch.stack(w[16::18]))
        self.value_ffn = ispin(torch.stack(w[17::18]))
        print("valueffn", self.value_ffn.shape)
        print(len(self.outputv), len(self.ln1w), offset)

        self.n_layer = len(self.ln1w)
        self.zero = torch.tensor(0).to(
            dtype=dtypein)
        self.one = torch.tensor(1).to(
            dtype=dtypein)
        self.f = torch.tensor([5]).to(
            dtype=dtypein)

        print(self.one.dtype)
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

    def toTensorFlowLayers(self) -> list[tensorflowrwkv.RWKVTFLayer]:
        l: list[tensorflowrwkv.RWKVTFLayer] = []
        for i in self.layerlist:
            layer = tensorflowrwkv.RWKVTFLayer(
                key=tf.convert_to_tensor(self.key[i][0].cpu().numpy()),
                receptance=tf.convert_to_tensor(
                    self.key[i][2].cpu().numpy()),
                value=tf.convert_to_tensor(self.key[i][1].cpu().numpy()),
                ln1w=tf.convert_to_tensor(self.ln1w[i].cpu().numpy()),
                ln1b=tf.convert_to_tensor(self.ln1b[i].cpu().numpy()),
                ln2w=tf.convert_to_tensor(self.ln2w[i].cpu().numpy()),
                ln2b=tf.convert_to_tensor(self.ln2b[i].cpu().numpy()),
                time_mix_r_ffn=tf.convert_to_tensor(
                    self.time_mix_r_ffn[i].cpu().numpy()),
                key_ffn=tf.convert_to_tensor(self.key_ffn[i].cpu().numpy()),
                kktk=tf.convert_to_tensor(self.kktk[i].cpu().numpy()),
                outputvv=tf.convert_to_tensor(self.outputv[i].cpu().numpy()),
                receptance_ffn=tf.convert_to_tensor(
                    self.receptance_ffn[i].cpu().numpy()),
                rrtr=tf.convert_to_tensor(self.rrtr[i].cpu().numpy()),
                time_decay=tf.convert_to_tensor(
                    self.time_decay[i].cpu().numpy()),
                time_first=tf.convert_to_tensor(
                    self.time_first[i].cpu().numpy()),
                time_mix_k_ffn=tf.convert_to_tensor(
                    self.time_mix_k_ffn[i].cpu().numpy()),
                value_ffn=tf.convert_to_tensor(
                    self.value_ffn[i].cpu().numpy()),
                vvtv=tf.convert_to_tensor(self.vvtv[i].cpu().numpy()),

            )
            l.append(layer)

        return l

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
            outputvv = self.stream(self.outputv)

            key_ffn = self.stream(self.key_ffn)
            receptance_ffn = self.stream(self.receptance_ffn)
            value_ffn = self.stream(self.value_ffn)

            statea = state[self.zero][self.uncint: self.cint]
            stateb = state[self.one][self.uncint: self.cint]
            statec = state[self.one+self.one][self.uncint: self.cint]
            stated = state[self.one+self.one+self.one][self.uncint: self.cint]

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
                atd = key[i]

                tmk = time_mix_k_ffn[i]
                tmr = time_mix_r_ffn[i]

                tmkw = key_ffn[i]
                tmrw = receptance_ffn[i]
                tmvw = value_ffn[i]

                # kmul1 = kmul11[i]

                xy = torch.layer_norm(
                    x, (ln1wa.shape), weight=ln1wa, bias=ln1ba)

                k = self.mv(atd[self.zero], xy + kktk[i]).to(torch.float64)

                v = self.mv(atd[self.one], xy + vvtv[i]).to(torch.float64)

                r = self.mv(atd[self.one + self.one], xy +
                            rrtr[i]).to(torch.float64)

                w = stateb[i].add(time_first[i].add(k).exp().mul(v)).div(statec[i].mul(r.exp()).add(time_first[i].add(k).add(
                    r).exp()).add(statec[i]).add(time_first[i].add(k).exp()))

                sxx = x.add(self.mv(outputvv[i], w.float()))

                statea[i] = xy
                stateb[i] = stateb[i].mul(time_decay[i].exp()).add(
                    k.exp().mul(v))  # ne33nd
                statec[i] = statec[i].mul(time_decay[i].exp()).add(k.exp())

                # return output, outstateA, outstateB, outstateC

                xx = torch.layer_norm(sxx, (ln2wa.shape),
                                      weight=ln2wa, bias=ln2ba)

                k = self.mv(tmkw, xx + tmk).relu().square()

                r = self.mv(tmrw, xx + tmr).exp()

                x = sxx.add(self.mv(tmvw, k).div(r.add(torch.ones_like(r))))

                stated[i] = xx

        return x, self.outputfunc(state, statea, stateb, statec, stated)


def empty_state(n_emb, layers, floatMode, device):
    state = torch.zeros(4, layers,
                        n_emb, device=device[0] if device[0] == "cpu" else "cuda", dtype=torch.float64)+0.01

    # state = torch.complex(state, torch.zeros_like(state))
    # for i in range(layers):
    #     state[5*i+4] -= 1e30
    # state = (*state,)
    return state


def empty_state_tf(n_emb, layers, floatMode, device):
    state = tf.zeros((4 * layers,
                      n_emb))+0.01

    # state = torch.complex(state, torch.zeros_like(state))
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
        setToProp(0)(w[0]), "cpu" if "cpu" in RunDevice[0] else "cuda", inttype=inttype)

    PostProcess = RWKV_POSTPROCESS(
        list(map(setToProp(0), w[2])), "cpu" if "cpu" in RunDevice[0] else "cuda", compatibility=compat, inttype=inttype)
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


def createRWKVTensorflowModel(Path, RunDevice=["cpu"], FloatMode=torch.float32, inttype=torch.int64, compat=False):
    preprocess, layers, postprocess, layersize = createRWKVModules(
        chunkSize=100, Path=Path, RunDevice=RunDevice, FloatMode=FloatMode, inttype=inttype, compat=compat)

    preprocess = tf.convert_to_tensor(preprocess.preProcess.cpu().numpy())
    layers = layers[0].toTensorFlowLayers()
    postprocess0 = tf.convert_to_tensor(postprocess.postProcess0.cpu().numpy())
    postprocess1 = tf.convert_to_tensor(postprocess.postProcess1.cpu().numpy())
    postprocess2 = tf.convert_to_tensor(postprocess.postProcess2.cpu().numpy())

    return tensorflowrwkv.RWKV(preprocess, [postprocess0, postprocess1, postprocess2], layers)
