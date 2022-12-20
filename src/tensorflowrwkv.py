import tensorflow as tf
import torch
import src.rwkvops


def RWKV(mpreprocess, mpostprocess, mlayers, mode="tensorflow"):

    ops = src.rwkvops.RwkvOpList[mode](
        len(mlayers), len(mlayers[0]["time_first"]))

    def layernorm(x, w, b):
        xee2 = x - ops.mean(x)

        x2 = ops.sqrt(ops.mean(xee2*xee2) + 0.000009999999747378752)

        return w*(xee2/x2) + b

    class RWKVTFLayer(ops.module):
        def __init__(self, dic):
            super(RWKVTFLayer, self).__init__()

            self.__dict__ = {k: ops.initTensor(v) for k, v in dic.items()}

            # for k, v in dic.items():
            #     print(len(v.shape), v.shape)
            # self.key = ops.initTensor(dic["key"])
            # self.receptance = ops.initTensor(dic["receptance"]).inv()
            # self.value = ops.initTensor(dic["value"])

            # self.ln1w = ops.initTensor(dic["ln1w"])
            # self.ln1b = ops.initTensor(dic["ln1b"])

            # self.ln2w = ops.initTensor(dic["ln2w"])
            # self.ln2b = ops.initTensor(dic["ln2b"])

            # self.time_mix_k_ffn = ops.initTensor(dic["time_mix_k_ffn"])
            # self.time_mix_r_ffn = ops.initTensor(dic["time_mix_r_ffn"])

            # self.key_ffn = ops.initTensor(dic["key_ffn"])
            # self.receptance_ffn = ops.initTensor(dic["receptance_ffn"])
            # self.value_ffn = ops.initTensor(dic["value_ffn"])

            # self.kktk = ops.initTensor(dic["kktk"])
            # self.vvtv = ops.initTensor(dic["vvtv"])
            # self.rrtr = ops.initTensor(dic["rrtr"])

            # self.time_first = ops.initTensor(dic["time_first"])
            # self.time_decay = ops.initTensor(dic["time_decay"])

            # self.outputvv = ops.initTensor(dic["outputvv"])

        @ ops.layerdef
        def forward(self, x, statea, stateb, statec, stated):
            xy = layernorm(x, self.ln1w, self.ln1b)

            k = ops.exp(ops.matvec(self.key, (xy+self.kktk*statea)))

            v = ops.matvec(self.value, (xy+self.vvtv*statea))

            r = ops.exp(ops.matvec(
                self.receptance, (xy+self.rrtr*statea))) + 1

            w = stateb + ops.exp(self.time_first)*k*v
            d = statec*r+ops.exp(self.time_first)*k*r

            mvv = ops.matvec(self.outputvv, w/(d+0.001))
            sxx = x + mvv

            aaa = xy

            bbb = stateb * ops.exp(self.time_decay) + k * v  # ne33nd
            ccc = statec * ops.exp(self.time_decay) + k

            # return output, outstateA, outstateB, outstateC

            xx = layernorm(sxx, self.ln2w, self.ln2b)

            kma = ops.matvec(self.key_ffn, (xx +
                                            self.time_mix_k_ffn * stated))
            km = ops.relu(kma)

            rt = ops.exp(ops.matvec(self.receptance_ffn,
                                    (xx + self.time_mix_r_ffn * stated))) + 1

            x = sxx + ops.matvec(self.value_ffn, km*km)/rt

            ddd = xx

            # print(aaa.shape, bbb.shape, ccc.shape, ddd.shape)

            return x, aaa, bbb, ccc, ddd

    class RWKVTFPre(ops.module):
        def __init__(self, preprocess):
            super(RWKVTFPre, self).__init__()
            self.preprocess = ops.initTensor(preprocess)

        @ ops.prefunc
        def forward(self, x):
            return self.preprocess[x[0]]

    class RWKVTFPost(ops.module):
        def __init__(self, postprocess):
            super(RWKVTFPost, self).__init__()
            self.postprocess0 = ops.initTensor(postprocess[0])
            self.postprocess1 = ops.initTensor(postprocess[1])
            self.postprocess2 = ops.initTensor(postprocess[2])

        @ ops.postfunc
        def forward(self, x):
            return ops.matvec(self.postprocess2, layernorm(x, self.postprocess0,
                                                           self.postprocess1))

    class myRWKV(ops.module):
        @ ops.initfunc
        def __init__(self, preprocess, postprocess, layers):
            super(myRWKV, self).__init__()
            self.preprocess = RWKVTFPre(preprocess)

            self.mylayers: list[RWKVTFLayer] = list(map(
                RWKVTFLayer, layers))

            self.postprocess = RWKVTFPost(postprocess)

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
            return x, ops.stack(ot, 0)

    return myRWKV(mpreprocess, mpostprocess, mlayers), ops.emptyState
