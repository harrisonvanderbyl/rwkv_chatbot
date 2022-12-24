from functools import reduce
import tensorflow as tf
import torch
import src.rwkvops

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


def RWKV(Path, mode="tensorflow", *args, **kwargs):

    n_layer = 0

    with torch.no_grad():
        w: Dict[str, torch.Tensor] = torch.load(
            Path, map_location="cpu")
        # refine weights and send to correct device
        keys = list(w.keys())
        for x in keys:
            if '.time_' in x:
                w[x] = w[x].squeeze()

            if '.time_decay' in x:
                w[x] = w[x].double()
                w[x] = w[x].clamp(-5, 5)
                w[x] = torch.exp(-torch.exp(w[x]))

            if 'receptance.weight' in x:
                w[x] = -w[x]

            w[x].requires_grad = False
            w[x] = w[x].to(dtype=torch.bfloat16)
            try:
                if (int(x.split('.')[1])+1 > n_layer):
                    n_layer = int(x.split('.')[1])+1
            except:
                pass

    # store weights in self.w

        keys = list(w.keys())

        preprocess = []
        for x in tqdm(range(len(w["emb.weight"]))):
            preprocess = preprocess + [torch.layer_norm(w["emb.weight"][x], (w["blocks.0.ln0.weight"].shape[0],),
                                                        weight=w["blocks.0.ln0.weight"], bias=w["blocks.0.ln0.bias"])]

        for x in range(n_layer):
            w[f"blocks.{x}.att.key.weight"] *= w[f"blocks.{x}.att.time_mix_k"]
            w[f"blocks.{x}.att.value.weight"] *= w[f"blocks.{x}.att.time_mix_v"]
            w[f"blocks.{x}.att.receptance.weight"] *= w[f"blocks.{x}.att.time_mix_r"]

            w[f"blocks.{x}.ffn.key.weight"] *= w[f"blocks.{x}.ffn.time_mix_k"]
            w[f"blocks.{x}.ffn.receptance.weight"] *= w[f"blocks.{x}.ffn.time_mix_r"]

            w[f"blocks.{x}.att.time_mix_k"] = 1 / \
                w[f"blocks.{x}.att.time_mix_k"] - 1
            w[f"blocks.{x}.att.time_mix_v"] = 1 / \
                w[f"blocks.{x}.att.time_mix_v"] - 1
            w[f"blocks.{x}.att.time_mix_r"] = 1 / \
                w[f"blocks.{x}.att.time_mix_r"] - 1

            w[f"blocks.{x}.ffn.time_mix_k"] = 1 / \
                w[f"blocks.{x}.ffn.time_mix_k"] - 1
            w[f"blocks.{x}.ffn.time_mix_r"] = 1 / \
                w[f"blocks.{x}.ffn.time_mix_r"] - 1

    # garbage collect

    gc.collect()
    torch.cuda.empty_cache()

    ops = src.rwkvops.RwkvOpList[mode](
        n_layer, len(w[f"blocks.0.ffn.time_mix_k"]), *args, **kwargs)

    class RWKVTFLayer(ops.module):
        def __init__(self, x):
            super(RWKVTFLayer, self).__init__()

            # self.__dict__ = {k: ops.initTensor(v) for k, v in dic.items()}

            # for k, v in dic.items():
            #     print(len(v.shape), v.shape)
            self.ln1w = ops.initTensor(w[f"blocks.{x}.ln1.weight"])
            self.ln1b = ops.initTensor(w[f"blocks.{x}.ln1.bias"])
            self.ln2w = ops.initTensor(w[f"blocks.{x}.ln2.weight"])
            self.ln2b = ops.initTensor(w[f"blocks.{x}.ln2.bias"])
            self.time_decay = ops.initTensor(w[f"blocks.{x}.att.time_decay"])
            self.time_first = ops.initTensor(w[f"blocks.{x}.att.time_first"])
            self.kktk = ops.initTensor(w[f"blocks.{x}.att.time_mix_k"])
            self.vvtv = ops.initTensor(w[f"blocks.{x}.att.time_mix_v"])
            self.rrtr = ops.initTensor(w[f"blocks.{x}.att.time_mix_r"])
            self.key = ops.initTensor(w[f"blocks.{x}.att.key.weight"])
            self.value = ops.initTensor(w[f"blocks.{x}.att.value.weight"])
            self.receptance = ops.initTensor(
                w[f"blocks.{x}.att.receptance.weight"])
            self.outputvv = ops.initTensor(w[f"blocks.{x}.att.output.weight"])
            self.time_mix_k_ffn = ops.initTensor(
                w[f"blocks.{x}.ffn.time_mix_k"])
            self.time_mix_r_ffn = ops.initTensor(
                w[f"blocks.{x}.ffn.time_mix_r"])
            self.key_ffn = ops.initTensor(w[f"blocks.{x}.ffn.key.weight"])
            self.receptance_ffn = ops.initTensor(
                w[f"blocks.{x}.ffn.receptance.weight"])
            self.value_ffn = ops.initTensor(w[f"blocks.{x}.ffn.value.weight"])

        @ ops.layerdef
        def forward(self, x, statea, stateb, statec, stated):
            xy = ops.layernorm(x, self.ln1w, self.ln1b)

            k = ops.exp(ops.matvec(self.key, (xy+self.kktk*statea)))

            v = ops.matvec(self.value, (xy+self.vvtv*statea))

            td = self.time_decay
            tf = ops.exp(self.time_first)

            w = stateb + k * v * tf
            d = (statec + k * tf)

            r = ops.exp(ops.matvec(
                self.receptance, (xy+self.rrtr*statea))) + 1

            wrd = (w/(r*d))

            if (wrd.isnan().any()):
                print("wrd is nan")
                print([w.isnan().any(), r.isnan().any(), d.isnan().any()])
                exit()

            mvv = ops.matvec(self.outputvv, wrd)

            if (mvv.isnan().any()):
                print("mvv is nan")
                exit()

            sxx = x + mvv

            aaa = xy
            bbb = stateb * td + k * v
            ccc = statec * td + k

            ddd = ops.layernorm(sxx, self.ln2w, self.ln2b)

            km = ops.relu(ops.matvec(self.key_ffn, (ddd +
                                                    self.time_mix_k_ffn * stated)))

            rt = ops.exp(ops.matvec(self.receptance_ffn,
                                    (ddd + self.time_mix_r_ffn * stated))) + 1

            x = sxx + ops.matvec(self.value_ffn, km*km)/rt

            return x, aaa, bbb, ccc, ddd

    class RWKVTFPre(ops.module):
        def __init__(self):
            super(RWKVTFPre, self).__init__()
            self.preprocess = ops.stack(list(map(ops.initTensor, preprocess)))

        @ ops.prefunc
        def forward(self, x):
            return self.preprocess[x[0]]

    class RWKVTFPost(ops.module):
        def __init__(self):
            super(RWKVTFPost, self).__init__()

            self.postprocess0 = ops.initTensor(w["ln_out.weight"])
            self.postprocess1 = ops.initTensor(w["ln_out.bias"])
            self.postprocess2 = ops.initTensor(w["head.weight"])

        @ ops.postfunc
        def forward(self, x):
            return ops.matvec(self.postprocess2, ops.layernorm(x, self.postprocess0,
                                                               self.postprocess1))

    class myRWKV(ops.module):
        @ ops.initfunc
        def __init__(self):
            super(myRWKV, self).__init__()
            self.preprocess = RWKVTFPre()

            self.mylayers: list[RWKVTFLayer] = []
            for i in range(n_layer):
                self.mylayers.append(RWKVTFLayer(i))

            self.postprocess = RWKVTFPost()

        @ ops.mainfunc
        def forward(self, x, state):

            x = self.preprocess.forward(x)

            statea = state[0::4]
            stateb = state[1::4]
            statec = state[2::4]
            stated = state[3::4]

            ot = []

            # print("start", len(self.mylayers))

            for i, l in list(enumerate(self.mylayers)):
                x, aaa, bbb, ccc, ddd = l.forward(
                    x, statea[i], stateb[i], statec[i], stated[i])
                ot = ot + [aaa, bbb, ccc, ddd]

            x = self.postprocess.forward(x)
            # print(len(ot))
            return x, ops.stack(ot)

    return ops.postProcessModule(myRWKV()), ops.emptyState
