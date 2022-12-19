import tensorflow as tf
import torch


def RWKV(mpreprocess, mpostprocess, mlayers, mode="tensorflow"):
    if mode == "tensorflow":
        initTensor = tf.convert_to_tensor
        sqrt = tf.sqrt
        mean = tf.reduce_mean
        def relu(x): return tf.maximum(x, tf.zeros_like(x))
        exp = tf.exp
        stack = tf.stack
        matvec = tf.linalg.matvec

        # module def
        module = tf.Module

        # tensorflow function defs
        layerdef = tf.function(
            input_signature=5*[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        mainFunc = tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.int32), tf.TensorSpec(
            shape=[4*len(mlayers), len(mlayers[0]["time_decay"])], dtype=tf.float32)])
        prefunc = tf.function(
            input_signature=[tf.TensorSpec(shape=[1], dtype=tf.int32)])
        emptyState = tf.zeros(
            [4*len(mlayers), len(mlayers[0]["time_decay"])], dtype=tf.float32)+0.01

    if mode == "pytorch":
        initTensor = torch.tensor
        sqrt = torch.sqrt
        mean = torch.mean
        relu = torch.relu
        exp = torch.exp
        stack = torch.stack
        matvec = torch.mv

        # module def
        module = torch.nn.Module

        # pytorch function defs
        def layerdef(x): return x
        def mainFunc(x): return x
        def prefunc(x): return x
        emptyState = torch.zeros(12*4, 768
                                 [4*len(mlayers), len(mlayers[0]["time_decay"])], dtype=torch.float32)+0.01

    def layernorm(x, w, b):
        xee2 = x - mean(x)

        x2 = sqrt(mean(xee2*xee2) + 0.000009999999747378752)

        return w*(xee2/x2) + b

    class RWKVTFLayer(module):
        def __init__(self, dic):
            self.key = initTensor(dic["key"])
            self.receptance = initTensor(dic["receptance"])
            self.value = initTensor(dic["value"])

            self.ln1w = initTensor(dic["ln1w"])
            self.ln1b = initTensor(dic["ln1b"])

            self.ln2w = initTensor(dic["ln2w"])
            self.ln2b = initTensor(dic["ln2b"])

            self.time_mix_k_ffn = initTensor(dic["time_mix_k_ffn"])
            self.time_mix_r_ffn = initTensor(dic["time_mix_r_ffn"])

            self.key_ffn = initTensor(dic["key_ffn"])
            self.receptance_ffn = initTensor(dic["receptance_ffn"])
            self.value_ffn = initTensor(dic["value_ffn"])

            self.kktk = initTensor(dic["kktk"])
            self.vvtv = initTensor(dic["vvtv"])
            self.rrtr = initTensor(dic["rrtr"])

            self.time_first = initTensor(dic["time_first"])
            self.time_decay = initTensor(dic["time_decay"])

            self.outputvv = initTensor(dic["outputvv"])

        @layerdef
        def forward(self, x, statea, stateb, statec, stated):
            xy = layernorm(x, self.ln1w, self.ln1b)

            k = exp(matvec(self.key, (xy+self.kktk*statea)))

            v = matvec(self.value, (xy+self.vvtv*statea))

            r = exp(matvec(
                self.receptance, (xy+self.rrtr*statea))) + 1

            w = stateb + exp(self.time_first)*k*v
            d = statec*r+exp(self.time_first)*k*r

            mvv = matvec(self.outputvv, w/(d+0.001))
            sxx = x + mvv

            aaa = xy

            bbb = stateb * exp(self.time_decay) + k * v  # ne33nd
            ccc = statec * exp(self.time_decay) + k

            # return output, outstateA, outstateB, outstateC

            xx = layernorm(sxx, self.ln2w, self.ln2b)

            kma = matvec(self.key_ffn, (xx +
                                        self.time_mix_k_ffn * stated))
            km = relu(kma)

            rt = exp(matvec(self.receptance_ffn,
                            (xx + self.time_mix_r_ffn * stated))) + 1

            x = sxx + matvec(self.value_ffn, km*km)/rt

            ddd = xx

            # print(aaa.shape, bbb.shape, ccc.shape, ddd.shape)

            return x, aaa, bbb, ccc, ddd

    class RWKVTFPre(module):
        def __init__(self, preprocess):
            self.preprocess = initTensor(preprocess)

        @prefunc
        def forward(self, x):
            return self.preprocess[x[0]]

    class RWKVTFPost(module):
        def __init__(self, postprocess):
            self.postprocess0 = initTensor(postprocess[0])
            self.postprocess1 = initTensor(postprocess[1])
            self.postprocess2 = initTensor(postprocess[2])

        def forward(self, x):
            return matvec(self.postprocess2, layernorm(x, self.postprocess0,
                                                       self.postprocess1))

    class myRWKV(module):

        def __init__(self, preprocess, postprocess, layers):
            super(myRWKV, self).__init__()
            self.preprocess = RWKVTFPre(preprocess)

            self.mylayers: list[RWKVTFLayer] = list(map(
                RWKVTFLayer, layers))

            self.postprocess = RWKVTFPost(postprocess)

        @mainFunc
        def forward(self, x, state):

            x = self.preprocess.forward(x)

            statea = state[0::4]
            stateb = state[1::4]
            statec = state[2::4]
            stated = state[3::4]

            ot = []

            # print("start", len(self.mylayers))

            for i, l in enumerate(self.mylayers):
                x, aaa, bbb, ccc, ddd = l.forward(
                    x, statea[i], stateb[i], statec[i], stated[i])
                ot = ot + [aaa, bbb, ccc, ddd]

            x = self.postprocess.forward(x)
            # print(len(ot))
            return x, stack(ot, 0)

    return myRWKV(mpreprocess, mpostprocess, mlayers), emptyState
