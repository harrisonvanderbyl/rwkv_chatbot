
import gc
from urllib import request
import inquirer
import numpy as np
import os
from typing import Dict as dict
import torch
import http.server
import json
import socketserver
from scipy.special import softmax

KLIMIT = 30
KLIMIT16 = 11

# allow tf32
torch.backends.cuda.matmul.allow_tf32 = True


def notimplemented(*args):
    raise "not implemented"


def npsample(ozut, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
    try:
        ozut = ozut.numpy()
    except:
        try:
            ozut = ozut.cpu().numpy()
        except:
            ozut = np.array(ozut)
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # turn to float if is half and cpu
    probs = softmax(ozut, axis=-1)

    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(
        cumulative_probs > top_p_usual)])
    probs[probs < cutoff] = 0
    if temp != 1.0:
        probs = pow(probs, 1.0 / temp)
    probs = probs / np.sum(probs, axis=0)
    mout = np.random.choice(a=len(probs), p=probs)
    return mout


class RWKVOPS():
    def __init__(self, layers, embed):
        print("init RWKVOPS, from super")
        self.initTensor: notimplemented
        self.initCpuTensor = lambda x: self.initTensor(x)
        self.sqrt: notimplemented
        self.mean: notimplemented
        self.relu: notimplemented
        self.exp: notimplemented
        self.add = lambda x, y: x+y
        self.divide = lambda x, y: x/y
        self.multiply = lambda x, y: x*y
        self.subtract = lambda x, y: x-y
        self.stack: notimplemented
        self.matvec: notimplemented
        self.layernorm: notimplemented
        self.lerp: notimplemented
        self.postProcessTensor = lambda x: x
       # module def
        self.module: notimplemented
        self.log: notimplemented
        self.minimum: notimplemented
        self.klimit: notimplemented
       # tensorflow function defs
        self.initfunc: notimplemented
        self.layerdef: notimplemented
        self.mainfunc: notimplemented
        self.emptyState: notimplemented
        self.logistical = lambda x: 1 / (self.exp(x) + 1)
        self.postProcessModule = lambda x: x

        self.sample = npsample

        # typing, set as any
        self.tensorDef = None


class RWKVTFOps(RWKVOPS):
    def __init__(self, layers, embed, useGPU: bool = None):
        try:
            import tensorflow as tf
        except:
            inst = inquirer.confirm(
                "Tensorflow not installed, do you want to install it?")
            if inst:
                os.system("pip3 install tensorflow")
                import tensorflow as tf
        if (not (inquirer.confirm("Do you want to use GPU?") if useGPU is None else useGPU)):
            tf.config.experimental.set_visible_devices([], "GPU")
        tf.config.optimizer.set_jit(True)
        tf.config.optimizer.set_experimental_options(
            {"auto_mixed_precision": True})

        super(RWKVTFOps, self).__init__(layers, embed)
        self.initTensor = lambda x: tf.convert_to_tensor(
            x.float().cpu().numpy())
        self.sqrt = tf.sqrt
        self.mean = tf.reduce_mean
        self.relu = lambda x: tf.maximum(x, tf.zeros_like(x))
        self.minimum = tf.minimum
        self.exp = tf.exp
        self.stack = tf.stack
        self.matvec = tf.linalg.matvec
        self.klimit = tf.convert_to_tensor(
            [KLIMIT]*embed, dtype=tf.float32
        )
        self.log = tf.math.log
        self.lerp = lambda x, y, z: x*(1-z)+y*z
       # module def
        self.module = tf.Module

       # tensorflow function defs
        self.initfunc = lambda x: x
        self.layerdef = tf.function(
            input_signature=5*[tf.TensorSpec(shape=[None], dtype=tf.float32)]+[tf.TensorSpec(dtype=tf.int64, shape=None)])
        self.mainfunc = tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.int32), tf.TensorSpec(
            shape=[4*layers, embed], dtype=tf.float32)])
        self.emptyState = tf.zeros([4*layers, embed], dtype=tf.float32)+0.01

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln


class RWKVTFExport(RWKVTFOps):
    def __init__(self, layers, embed,  exports=None):
        super(RWKVTFExport, self).__init__(layers, embed)
        import tensorflow as tf
        self.module = tf.keras.Model
        path = f"tfdist/rwkv-{layers}-{embed}/"

        def save(x):
            x([0], self.emptyState)
            try:
                try:
                    os.mkdir("tfdist")
                except:
                    pass
                os.mkdir(path)
            except:
                pass

            q = exports if exports is not None else inquirer.checkbox(message="What to export?", choices=[
                "savedmodel32", "tflite32", "tflite16"])

            if "savedmodel32" in q:
                try:
                    os.mkdir(path+"sm")
                except:
                    pass

                tf.keras.models.save_model(x, path+"sm/whole")

            if "tflite32" in q:
                try:
                    os.mkdir(path+"tflite32")
                except:
                    pass

                converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    [x.forward.get_concrete_function()])
                tflite_model = converter.convert()
                open(f"model-{layers}-{embed}-32.tflite",
                     "wb").write(tflite_model)

            if "tflite16" in q:
                try:
                    os.mkdir(path+"tflite16")
                except:
                    pass

                converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    [x.forward.get_concrete_function()])
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                tflite_model = converter.convert()
                open(f"model-{layers}-{embed}-16.tflite",
                     "wb").write(tflite_model)
            exit()
        self.postProcessModule = save


class RWKVNumpyOps(RWKVOPS):
    def __init__(self, layers, embed):
        super().__init__(layers, embed)
        self.initTensor = lambda x: x.float().cpu().numpy()
        self.sqrt = lambda x: np.sqrt(x)
        self.mean = lambda x: np.mean(x)
        self.relu = lambda x: np.maximum(x, 0)
        self.exp = lambda x: np.exp(x)
        self.stack = lambda x: x
        self.matvec = np.matmul
        self.lerp = lambda x, y, z: x*(1-z) + y*(z)
        self.minimum = lambda x, y: np.minimum(x, y)
        self.klimit = [KLIMIT] * embed
        # module def
        self.module = object
        self.log = np.log

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b
        self.layernorm = ln
        self.emptyState = [[0.01]*embed]*4*layers


class RWKVJaxOps(RWKVOPS):
    def __init__(self, layers, embed):
        from jax import numpy as npjax
        super().__init__(layers, embed)
        self.initTensor = lambda x: npjax.array(x.float().cpu().numpy())
        self.sqrt = lambda x: npjax.sqrt(x)
        self.mean = lambda x: npjax.mean(x)
        self.relu = lambda x: npjax.maximum(x, 0)
        self.exp = lambda x: npjax.exp(x)
        self.stack = lambda x: x
        self.matvec = npjax.matmul
        self.lerp = lambda x, y, z: x*(1-z) + y*(z)
        self.minimum = lambda x, y: npjax.minimum(x, y)
        self.klimit = npjax.array([KLIMIT] * embed)
        # module def
        self.module = object
        self.log = npjax.log

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x
        # in postfunc, convert to numpy

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln
        self.emptyState = npjax.array([[0.01]*embed]*4*layers)


class RWKVJaxIreeOps(RWKVJaxOps):
    def __init__(self, layers, embed):
        RWKVJaxOps.__init__(self, layers, embed)
        from iree.jax import Program
        from jax import numpy as npjax

        self.module = Program
        self.layerdef = Program.kernel
        # annotate the function

        self.tensorDef = Program.like(self.initTensor(torch.ones((embed))))

        self.emptyState = Program.like(self.emptyState)

        # self.postProcessModule()


def torchsample(ozut: torch.LongTensor, temp=1.0, top_p_usual=0.8) -> int:
    # do it in pytorch

    probs = torch.softmax(ozut, dim=-1)
    sorted_probs, indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = sorted_probs[torch.argmax(
        cumulative_probs[cumulative_probs > top_p_usual])]
    probs[probs < cutoff] = 0
    if temp != 1.0:
        probs = torch.pow(probs, 1.0 / temp)
    probs = probs / torch.sum(probs, dim=-1)
    mout = torch.multinomial(probs, 1)
    return mout.cpu()


class RWKVPTOps(RWKVOPS):

    def __init__(self, layers, embed, dtype=None):
        RWKVOPS.__init__(self, layers, embed)
        q = [inquirer.List(
            'type',
            message="Load model with which dtype?",
            choices=[torch.bfloat16, torch.float16, torch.float32, torch.float64])]

        if dtype is None:
            a = inquirer.prompt(q)
            dtype = a['type']
        self.dtype = dtype
        # self.sample = torchsample

        def initTensor(x):
            result = x.to(dtype=self.dtype)

            return result

        self.initTensor = initTensor
        self.initCpuTensor = lambda x: self.initTensor(x).cpu()
        self.klimit = torch.tensor(
            [KLIMIT] * embed).to(dtype=self.dtype)
        self.minimum = torch.minimum
        self.sqrt = torch.sqrt
        self.mean = torch.mean
        self.relu = torch.relu
        self.stack = lambda x: x
        self.matvec = torch.mv
        # safe log
        self.log = lambda x: torch.complex(x, torch.zeros_like(x)).log()

        self.exp = lambda x: torch.exp(x).to(dtype=self.dtype)
        self.lerp = torch.lerp

        # module def
        self.module = torch.nn.Module

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x

        # self.postProcessModule = ppm

        def layernorm(x, w, b) -> torch.Tensor:

            return torch.layer_norm(x, w.shape, w, b)
        self.layernorm = layernorm
        self.emptyState = torch.zeros(
            4*layers, embed, dtype=self.dtype)+0.0


class RWKVPoptorchOps(RWKVPTOps):
    def __init__(self, layers, embed, *args):
        super().__init__(layers, embed, *args)
        try:
            import poptorch
        except:
            raise ImportError("poptorch not installed")
        self.postProcessModule = poptorch.inferenceModel


class RWKVPTCompatOps(RWKVPTOps):
    def __init__(self, layers, embed, *args):
        RWKVPTOps.__init__(self, layers, embed, *args)
        self.relu = lambda x: torch.max(x, torch.zeros_like(x))
        self.matvec = lambda x, y: torch.sum(x*y, dim=1)

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln


class RWKVCudaOps(RWKVPTOps):
    def __init__(self, layers, embed, *args, useGPU=None, runtimedtype=None, **kwargs):
        super().__init__(layers, embed, *args, **kwargs)

        useGPU = inquirer.confirm(
            "Use GPU?", default=True) if useGPU is None else useGPU

        self.useGPU = useGPU

        if not useGPU:
            return

        runtimedtype = inquirer.prompt([inquirer.List(
            'type',
            message="Dtype for non-matrix ops:",
            choices=[torch.bfloat16, torch.float32, torch.float64])])['type'] if runtimedtype is None else runtimedtype

        self.exp = lambda x: torch.exp(x).to(dtype=runtimedtype)

        self.initTensor = lambda x: x.to(dtype=self.dtype if len(
            x.shape) == 2 else runtimedtype, device='cuda')
        self.initCpuTensor = self.initTensor  # could be used for offload

        self.klimit = self.klimit.to(dtype=runtimedtype, device='cuda')

        self.matvec = lambda x, y: x.mv(
            y.to(self.dtype)).to(runtimedtype)

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln

        self.emptyState = torch.zeros(
            4*layers, embed, dtype=runtimedtype, device="cuda")+0.01


class RWKVPTTSExportOps(RWKVCudaOps):
    def __init__(self, layers, embed, *args, includeSampler=None):
        super().__init__(layers, embed, *args)
        self.stack = lambda x: torch.stack(x)

        includeSampler = inquirer.confirm(
            "Include sampler?", default=True) if includeSampler is None else includeSampler

        if includeSampler:
            self.sample = torchsample
            self.postProcessTensor = lambda x: self.sample(
                x.float().cpu(), torch.tensor(1), torch.tensor(0.9))

        def exportTorchScript(x):
            torch.jit.save(torch.jit.trace(
                x, (torch.LongTensor([0]), self.emptyState), check_trace=False, strict=False), f"model-{layers}-{embed}-{'sampler' if includeSampler else 'logits'}-{'gpu' if self.useGPU else 'cpu'}-{self.dtype}.pt")
            exit()
        self.postProcessModule = exportTorchScript


class RWKVCudaDeepspeedOps(RWKVCudaOps):
    def __init__(self, layers, embed, *args):
        super().__init__(layers, embed, *args)

        try:
            import deepspeed
        except:
            raise ImportError("deepspeed not installed")

        self.postProcessModule = lambda x: deepspeed.init_inference(
            x, replace_method='auto', replace_with_kernel_inject=True).module


def ConvertUin8MatToUint4Mat(x):
    x = x.reshape(x.shape[0], x.shape[1]//2, 2)
    x = x[:, :, 0]*16+x[:, :, 1]
    return x


def DeconvertUint4MatToUint8Mat(x):
    mx = torch.zeros(x.shape[0], x.shape[1]*2,
                     dtype=torch.uint8, device=x.device)
    mx[:, ::2] = x//16
    mx[:, 1::2] = x % 16
    return mx


def QuantizeMatrix(x, runtimeDtype, device, uint4=False):
    rang = 255 if not uint4 else 15
    ran, mini = (x.max(0)[0]-x.min(0)[0])/rang,  x.min(0)[0]
    x = x.double()
    x = ((x-mini)/ran)

    x = x.to(
        dtype=torch.uint8, non_blocking=True, device=device)

    # if uint4:
    #     x = ConvertUin8MatToUint4Mat(x)

    return x, ran.to(runtimeDtype).to(device=device), mini.to(runtimeDtype).to(device=device)


def QuantizedMatVec(x, y, runtimedtype, uint4=False):
    rx, spread, zpoint = x
    yy = y*spread

    # if uint4:
    #     rx = DeconvertUint4MatToUint8Mat(rx)

    rx = rx.to(dtype=runtimedtype)

    xmain = rx.matmul(yy.reshape(yy.shape[0], -1, 1)).sum(0).squeeze()

    # print(xmain.shape)

    return xmain + torch.tensordot(zpoint, y)


class RWKVCudaQuantOps(RWKVPTOps):
    def __init__(self, layers, embed, *args, runtimedtype=None, uint4=False, useGPU=None, chunksize=None):
        super().__init__(layers, embed, torch.bfloat16)
        import matplotlib.pyplot as plt
        dev = 'cuda' if (inquirer.confirm(
            "Use GPU?", default=True) if useGPU is None else useGPU) else 'cpu'

        runtimedtype = inquirer.prompt([inquirer.List(
            'type',
            message="Dtype for operations:",
            choices=[torch.bfloat16, torch.float16, torch.float32, torch.float64])])['type'] if runtimedtype is None else runtimedtype

        uint4 = inquirer.confirm(
            "Use uint4?", default=False) if uint4 is None else uint4

        chunksize = inquirer.prompt([inquirer.List(
            'chunksize',
            message="Chunksize(Trade speed for accuracy):",
            choices=[1, 2, 4, 8, 16, 32, 64, 128, 256])])['chunksize'] if chunksize is None else chunksize

        def initTensor(x):
            if (len(x.shape) != 2):
                return x.to(dtype=runtimedtype, device=dev)

            splitmatrices = torch.chunk(x, chunksize, 1)
            xx = [QuantizeMatrix(x, runtimedtype, dev, uint4)
                  for x in splitmatrices]
            xxo = torch.stack([x[0] for x in xx])
            xx1 = torch.stack([x[1] for x in xx])
            xx2 = torch.stack([x[2] for x in xx])
            return xxo, xx1, xx2

        self.initTensor = initTensor
        self.initCpuTensor = self.initTensor

        self.postProcessModule = lambda x: x

        def matvec(x, y):
            splitVectors = y.reshape(chunksize, -1)
            return QuantizedMatVec(x, splitVectors, runtimedtype, uint4)

        self.matvec = matvec

        self.klimit = self.klimit.to(dtype=runtimedtype, device=dev)

        self.emptyState = torch.zeros(
            4*layers, embed, dtype=runtimedtype, device=dev)+0.01


class RWKVExportOnnxOps(RWKVCudaOps):
    def __init__(self, layers, embed, *args):
        base = inquirer.prompt([inquirer.List(
            'type',
            message="Base class for export:",
            choices=["Normal", "Compat"])])['type']

        if base == "Normal":
            base = RWKVCudaOps
        elif base == "Compat":
            base = RWKVPTCompatOps
        base.__init__(self, layers, embed, *args)
        path = f"rwkv-{layers}-{embed}-{torch.float32}.onnx"
        # super().__init__(layers, embed)
        self.stack = torch.stack

        onnxOpversion = inquirer.prompt([inquirer.List(
            'type',
            message="ONNX Opset version:",
            choices=[12, 13, 14, 15, 16, 17])])['type']

        def export(x):
            torch.onnx.export(x, ([0], self.emptyState),
                              path, opset_version=onnxOpversion)
        self.postProcessModule = export


class RWKVStreamBigOps(RWKVPTOps):
    def __init__(self, layers, embed, processDtype=torch.float32, storageDtype=torch.bfloat16, target=None):
        super().__init__(layers, embed, dtype=storageDtype)

        pinMem = inquirer.prompt([inquirer.Confirm(
            'type',
            message=f"Pin memory to cpu?",
            default=True)])['type']

        def pinmem(x):
            return x.pin_memory() if pinMem and x.device == "cpu" else x

        target = target if target is not None else float(
            input("Designate the amount of memory to allocate (in GB):"))
        self.initTensor = lambda x: pinmem(x.to(device='cpu' if len(x.shape) == 2 else "cuda", dtype=storageDtype if len(x.shape) == 2 else processDtype)) if (
            torch.cuda.max_memory_reserved(0)/1024/1024/1024) > target else x.to(dtype=storageDtype if len(x.shape) == 2 else processDtype).cuda()

        # for everything in self, if its a tensor, send to cuda

        self.initCpuTensor = self.initTensor
        self.klimit = self.klimit.cuda(non_blocking=True)
        self.matvec = lambda z, y: z.cuda(non_blocking=True).mv(
            y.to(storageDtype)).to(processDtype)
        self.emptyState = torch.zeros(
            4*layers, embed, dtype=processDtype, device="cuda")+0.01

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln


class RWKVSplitCudaOps(RWKVPTOps):
    def __init__(self, layers, embed, processDtype=torch.float32, storageDtype=torch.bfloat16, target=None):
        super().__init__(layers, embed, dtype=storageDtype)

        devices = inquirer.checkbox(
            'Which devices would you like to use?', choices=['cpu', 'cuda:0', 'cuda:1'])

        self.initTensor = lambda x: x.to(dtype=processDtype).cuda() if len(
            x.shape) == 1 else list(map(lambda zx: zx[1].to(device=devices[zx[0]], dtype=torch.float32 if "cpu" in devices[zx[0]] else torch.bfloat16), enumerate(list(x.chunk(len(devices), dim=1)))))
        self.initCpuTensor = self.initTensor

        # for everything in self, if its a tensor, send to cuda
        # self.matvec = lambda x, y: x.mv(y.to(torch.float16)).to(processDtype)
        self.emptyState = torch.zeros(
            4*layers, embed, dtype=processDtype, device="cuda")+0.01

        self.minimum = lambda x, y: torch.min(x, torch.ones_like(x)*KLIMIT)

        def matvec(matx, y):
            chunks = list(map(lambda xx: xx[1].to(
                device=devices[xx[0]], dtype=matx[xx[0]].dtype, non_blocking=True), enumerate(y.chunk(len(devices), dim=0))))
            res = matx[0].mv(chunks[0]).to(
                dtype=processDtype, device=y.device, non_blocking=True)
            for i in range(1, len(chunks)):
                res = res + matx[i].mv(chunks[i]).to(
                    dtype=processDtype, device=y.device, non_blocking=True)

            return res

        self.stack = lambda x: x

        self.matvec = matvec
        self.layernorm = lambda x, w, b: torch.layer_norm(
            x.to(device=w.device), w.shape, w, b)


class RWKVMobileOps(RWKVPTOps):
    def __init__(self, layers, embed, *args):
        super().__init__(layers, embed, *args)
        path = f"PTMobile/rwkv-{layers}-{embed}-{self.dtype}/"
        self.stack = torch.stack

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b
        self.layernorm = ln
        dtype = self.dtype

        def export(self):
            print("exporting")
            try:
                try:
                    os.mkdir("PTMobile")
                except:
                    pass
                os.mkdir(path)
            except:
                pass
            self.preprocess = torch.jit.trace(
                self.preprocess, (torch.zeros(1, dtype=torch.int32),))
            # torch.onnx.export(
            #     self.preprocess, (torch.zeros(1, dtype=torch.int32),), f"{path}pre.onnx")
            self.postprocess = torch.jit.trace(
                self.postprocess, (torch.zeros(embed, dtype=dtype),))
            # torch.onnx.export(
            #     self.postprocess, (torch.zeros(embed, dtype=dtype),), f"{path}post.onnx")
            for i, layer in enumerate(self.mylayers):
                self.mylayers[i] = torch.jit.trace(
                    layer, (torch.zeros(embed, dtype=dtype)+0.01, torch.zeros(embed, dtype=dtype)+0.01, torch.zeros(embed, dtype=dtype)+0.01, torch.zeros(embed, dtype=dtype)+0.01, torch.zeros(embed, dtype=dtype)+0.01))

                # torch.onnx.export(
                #     layer, (torch.zeros(embed, dtype=dtype)+0.01, torch.zeros(embed, dtype=dtype)+0.01, torch.zeros(embed, dtype=torch.float32)+0.01, torch.zeros(embed, dtype=torch.float32)+0.01, torch.zeros(embed, dtype=torch.float32)+0.01), f"{path}{i}.onnx")
            self.preprocess._save_for_lite_interpreter(f"{path}pre.ptl")
            self.postprocess._save_for_lite_interpreter(f"{path}post.ptl")
            for i, layer in enumerate(self.mylayers):
                layer._save_for_lite_interpreter(f"{path}{i}.ptl")

            return self
        self.postProcessModule = export

        self.mainfunc = lambda x: lambda self, r, * \
            args: x(self, torch.tensor(r).to(torch.int32), *args)


RwkvOpList = {
    "tensorflow(cpu/gpu)": RWKVTFOps,
    "pytorch(cpu/gpu)": RWKVCudaOps,
    "numpy(cpu)": RWKVNumpyOps,
    "jax(cpu/gpu/tpu)": RWKVJaxOps,
    "pytorch-deepspeed(gpu)": RWKVCudaDeepspeedOps,
    "pytorch-quant(gpu-8bit)": RWKVCudaQuantOps,
    "pytorch-stream(gpu-config-vram)": RWKVStreamBigOps,
    "pytorch-split(2xgpu)": RWKVSplitCudaOps,
    "export-torchscript": RWKVPTTSExportOps,
    "export-tensorflow": RWKVTFExport,
    "export-onnx": RWKVExportOnnxOps,
    "pytorch-compatibility(cpu/debug)": RWKVPTCompatOps,

    "poptorch(idk)": RWKVPoptorchOps,


}
