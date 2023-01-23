
import torch
from src.interopFileLoader import initTFLiteFile, initTorchScriptFile
from src.rwkvops import RwkvOpList as Backends
from src.rwkvMaster import RWKVMaster
import torch
import gc
from typing import Dict
from tqdm import tqdm
import inquirer
import os

# set torch threads to 8
torch.set_num_threads(8)


def RWKV(Path=None, mode=None, *args, **kwargs) -> RWKVMaster:

    if (Path == None):
        files = os.listdir()
        # filter by ending in .pth
        files = [f for f in files if f.endswith(
            ".pth") or f.endswith(".pt") or f.endswith(".tflite")]

        questions = [
            inquirer.List('file',
                          message="What model do you want to use?",
                          choices=files,
                          )]
        Path = inquirer.prompt(questions)["file"]

    if Path.endswith(".pt"):
        return initTorchScriptFile(Path)
    elif Path.endswith(".tflite"):
        return initTFLiteFile(Path)

    if mode is None:
        mode = inquirer.prompt([inquirer.List('mode',
                                              message="What inference backend do you want to use?",
                                              choices=Backends.keys(),
                                              )])["mode"]

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
                w[x] = torch.exp(-torch.exp(w[x].double())
                                 )

            if 'receptance.weight' in x:
                w[x] = -w[x]

            w[x].requires_grad = False

            try:
                if (int(x.split('.')[1])+1 > n_layer):
                    n_layer = int(x.split('.')[1])+1
            except:
                pass

    # store weights in self.w

        keys = list(w.keys())

        ops = Backends[mode](
            n_layer, len(w[f"blocks.0.ffn.time_mix_k"]), *args, **kwargs)

        for x in tqdm(list(w.keys())):
            if "emb.weight" in x:
                w[x] = ops.stack(list(map(lambda rrx: ops.initTensor(
                    rrx.squeeze()), w[x].split(1, 0))))
            else:
                w[x] = ops.initTensor(w[x])

        gc.collect()
        torch.cuda.empty_cache()

        class myRWKV(ops.module):
            postprocess0 = (w["ln_out.weight"])
            postprocess1 = (w["ln_out.bias"])
            postprocess2 = (w["head.weight"])
            emb = w["emb.weight"]
            emb1 = w["blocks.0.ln0.weight"]
            emb2 = w["blocks.0.ln0.bias"]
            ln1w = ops.stack(
                [w[f"blocks.{x}.ln1.weight"] for x in range(n_layer)])
            ln1b = ops.stack(
                [w[f"blocks.{x}.ln1.bias"] for x in range(n_layer)])
            ln2w = ops.stack(
                [w[f"blocks.{x}.ln2.weight"] for x in range(n_layer)])
            ln2b = ops.stack(
                [w[f"blocks.{x}.ln2.bias"] for x in range(n_layer)])
            time_decay = ops.stack([
                w[f"blocks.{x}.att.time_decay"] for x in range(n_layer)])
            time_first = ops.stack([
                w[f"blocks.{x}.att.time_first"] for x in range(n_layer)])
            kktk = ops.stack(
                [w[f"blocks.{x}.att.time_mix_k"] for x in range(n_layer)])
            vvtv = ops.stack(
                [w[f"blocks.{x}.att.time_mix_v"] for x in range(n_layer)])
            rrtr = ops.stack(
                [w[f"blocks.{x}.att.time_mix_r"] for x in range(n_layer)])
            key = ops.stack(
                [w[f"blocks.{x}.att.key.weight"] for x in range(n_layer)])
            value = ops.stack(
                [w[f"blocks.{x}.att.value.weight"] for x in range(n_layer)])
            receptance = ops.stack([
                w[f"blocks.{x}.att.receptance.weight"] for x in range(n_layer)])
            outputvv = ops.stack([
                w[f"blocks.{x}.att.output.weight"] for x in range(n_layer)])
            time_mix_k_ffn = ops.stack([
                w[f"blocks.{x}.ffn.time_mix_k"] for x in range(n_layer)])
            time_mix_r_ffn = ops.stack([
                w[f"blocks.{x}.ffn.time_mix_r"] for x in range(n_layer)])
            key_ffn = ops.stack(
                [w[f"blocks.{x}.ffn.key.weight"] for x in range(n_layer)])
            receptance_ffn = ops.stack([
                w[f"blocks.{x}.ffn.receptance.weight"] for x in range(n_layer)])
            value_ffn = ops.stack([
                w[f"blocks.{x}.ffn.value.weight"] for x in range(n_layer)])

            @ ops.initfunc
            def __init__(self):
                super(myRWKV, self).__init__()

                self.ops = ops

                # self.postprocess = RWKVTFPost()
            @ops.layerdef
            def doLayer(self, x, statea, stateb, statec, stated, xx):
                xy = ops.layernorm(x, self.ln1w[xx], self.ln1b[xx])

                kk = ops.matvec(
                    self.key[xx], ops.lerp(statea, xy, self.kktk[xx]))

                v = ops.matvec(self.value[xx], ops.lerp(
                    statea, xy, self.vvtv[xx]))

                r = ops.logistical(ops.matvec(
                    self.receptance[xx], ops.lerp(statea, xy, self.rrtr[xx])))

                kt = ops.exp(ops.minimum(
                    ops.add(kk, self.time_first[xx]), ops.klimit))
                k = ops.exp(ops.minimum(kk, ops.klimit))

                wrd = ops.divide(
                    ops.add(stateb, ops.multiply(kt, v)), ops.add(statec, kt))
                outb = ops.add(ops.multiply(
                    stateb, self.time_decay[xx]), ops.multiply(k, v))
                outc = ops.add(ops.multiply(statec, self.time_decay[xx]), k)

                mvv = ops.add(x, ops.matvec(
                    self.outputvv[xx], ops.multiply(r, wrd)))

                ddd = ops.layernorm(mvv, self.ln2w[xx], self.ln2b[xx])

                km = ops.relu(ops.matvec(self.key_ffn[xx], ops.lerp(
                    stated, ddd, self.time_mix_k_ffn[xx])))

                rt = ops.logistical(ops.matvec(self.receptance_ffn[xx], ops.lerp(
                    stated, ddd, self.time_mix_r_ffn[xx])))

                x = ops.add(mvv, ops.multiply(
                    ops.matvec(self.value_ffn[xx], km*km), rt))

                return x, xy, outb, outc, ddd

            @ ops.mainfunc
            def forward(self, x, state=None):

                if (state is None):
                    state = ops.emptyState

                x = ops.layernorm(
                    self.emb[x[-1]], self.emb1, self.emb2)

                statea = state[0::4]
                stateb = state[1::4]
                statec = state[2::4]
                stated = state[3::4]

                ot = []

                for i in range(n_layer):
                    x, aaa, bbb, ccc, ddd = self.doLayer(
                        x, statea[i], stateb[i], statec[i], stated[i], i)
                    ot = ot + [aaa, bbb, ccc, ddd]

                x = ops.matvec(self.postprocess2, ops.layernorm(x, self.postprocess0,
                                                                self.postprocess1))

                return ops.postProcessTensor(x), ops.stack(ot)

            # for keras stuff, ignore this if you are not using keras
            def call(self, *args, **kwds):
                del kwds["training"]
                return self.forward(*args, **kwds)

        model = ops.postProcessModule(myRWKV())
        emptyState = ops.emptyState
        initTensor = ops.initTensor

        ret = RWKVMaster(model, emptyState, initTensor, ops.sample)

        return ret
