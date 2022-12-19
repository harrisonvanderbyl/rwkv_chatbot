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


def process(x): return x.cpu().float().numpy()


class RWKV_LAYER(nn.Module):
    def __init__(self, w):
        super().__init__()

        self.ln1w = (torch.stack(w[0::18]))
        self.ln1b = (torch.stack(w[1::18]))
        self.ln2w = (torch.stack(w[2::18]))
        self.ln2b = (torch.stack(w[3::18]))
        self.time_decay = ((torch.stack(w[4::18])))
        self.time_first = ((torch.stack(w[5::18])))

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

        self.vvtv = (torch.stack(tv))
        self.kktk = (torch.stack(tk))
        self.rrtr = (torch.stack(tr))

        self.key = (torch.stack(mm))
        self.outputv = (torch.stack(w[12::18]))
        self.time_mix_k_ffn = (torch.stack(w[13::18]))
        self.time_mix_r_ffn = (torch.stack(w[14::18]))
        self.key_ffn = (torch.stack(w[15::18]))

        self.receptance_ffn = (-torch.stack(w[16::18]))
        self.value_ffn = (torch.stack(w[17::18]))

        gc.collect()
        torch.cuda.empty_cache()

    def toTensorFlowLayers(self):

        l: list[dict] = []
        for i in range(len(self.key)):
            layer = dict(
                key=process((self.key[i][0])),
                receptance=process((
                    self.key[i][2])),
                value=process((self.key[i][1])),
                ln1w=process((self.ln1w[i])),
                ln1b=process((self.ln1b[i])),
                ln2w=process((self.ln2w[i])),
                ln2b=process((self.ln2b[i])),
                time_mix_r_ffn=process((
                    self.time_mix_r_ffn[i])),
                key_ffn=process((self.key_ffn[i])),
                kktk=process((self.kktk[i])),
                outputvv=process((self.outputv[i])),
                receptance_ffn=process((
                    self.receptance_ffn[i])),
                rrtr=process((self.rrtr[i])),
                time_decay=process((
                    self.time_decay[i])),
                time_first=process((
                    self.time_first[i])),
                time_mix_k_ffn=process((
                    self.time_mix_k_ffn[i])),
                value_ffn=process((
                    self.value_ffn[i])),
                vvtv=process((self.vvtv[i])),

            )
            l.append(layer)

        return l


def createRWKVModel(Path, mode="tensorflow"):
    w = createTensors(Path[: -4])

    preprocess = w[0]

    postprocess = w[2]

    modelLayers = RWKV_LAYER(w[1])

    preprocess = process(preprocess)
    layers = modelLayers.toTensorFlowLayers()

    return tensorflowrwkv.RWKV(preprocess, list(map(process, postprocess)), layers, mode=mode)
