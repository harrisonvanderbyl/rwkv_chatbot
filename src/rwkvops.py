import tensorflow as tf
import torch


def notimplemented(*args):
    raise "not implemented"


class RWKVOPS():
    def __init__(self, layers, embed):
        self.initTensor: notimplemented
        self.sqrt: notimplemented
        self.mean: notimplemented
        self.relu: notimplemented
        self.exp: notimplemented
        self.stack: notimplemented
        self.matvec: notimplemented

       # module def
        self.module: notimplemented

       # tensorflow function defs
        self.layerdef: notimplemented
        self.mainFunc: notimplemented
        self.prefunc: notimplemented
        self.postfunc: notimplemented
        self.emptyState: notimplemented


class RWKVTFOps(RWKVOPS):
    def __init__(self, layers, embed):
        self.initTensor = tf.convert_to_tensor
        self.sqrt = tf.sqrt
        self.mean = tf.reduce_mean
        self.relu = lambda x: tf.maximum(x, tf.zeros_like(x))
        self.exp = tf.exp
        self.stack = tf.stack
        self.matvec = tf.linalg.matvec

       # module def
        self.module = tf.Module

       # tensorflow function defs
        self.layerdef = tf.function(
            input_signature=5*[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        self.mainFunc = tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.int32), tf.TensorSpec(
            shape=[4*layers, embed], dtype=tf.float32)])
        self.prefunc = tf.function(
            input_signature=[tf.TensorSpec(shape=[1], dtype=tf.int32)])
        self.postfunc = lambda x: x
        self.emptyState = tf.zeros([4*layers, embed], dtype=tf.float32)+0.01


class RWKVPTOps(RWKVOPS):
    def __init__(self, layers, embed):

        self.initTensor = torch.tensor
        self.sqrt = torch.sqrt
        self.mean = torch.mean
        self.relu = torch.relu
        self.exp = torch.exp
        self.stack = torch.stack
        self.matvec = torch.mv

        # module def
        self.module = torch.nn.Module

        # pytorch function defs
        self.layerdef = lambda x: x
        self.mainFunc = lambda x: x
        self.postfunc = lambda x: x
        self.prefunc = lambda x: x
        self.emptyState = torch.zeros(
            4*layers, embed, dtype=torch.float32)+0.01


class RWKVPTCompatOps(RWKVOPS):
    def __init__(self, layers, embed):

        self.initTensor = torch.tensor
        self.sqrt = torch.sqrt
        self.mean = torch.mean
        self.relu = lambda x: torch.max(x, torch.zeros_like(x))
        self.exp = torch.exp
        self.stack = torch.stack
        self.matvec = lambda x, y: torch.sum(x*y, dim=1)

        # module def
        self.module = torch.nn.Module

        # pytorch function defs
        self.layerdef = lambda x: x
        self.mainFunc = lambda x: x
        self.postfunc = lambda x: x
        self.prefunc = lambda x: x
        self.emptyState = torch.zeros(
            4*layers, embed, dtype=torch.float32)+0.01


class RWKVCudaOps(RWKVOPS):
    def __init__(self, layers, embed):

        self.initTensor = lambda x: torch.tensor(x, device='cuda')
        self.sqrt = torch.sqrt
        self.mean = torch.mean
        self.relu = torch.relu
        self.exp = torch.exp
        self.stack = torch.stack
        self.matvec = torch.mv

        # module def
        self.module = torch.nn.Module

        # pytorch function defs
        self.layerdef = lambda x: x
        self.mainFunc = lambda x: x
        self.postfunc = lambda x: x
        self.prefunc = lambda x: x
        self.emptyState = torch.zeros(
            4*layers, embed, dtype=torch.float32, device="cuda")+0.01


class RWKVStreamOps(RWKVOPS):
    def __init__(self, layers, embed):

        self.initTensor = lambda x: torch.tensor(
            x, device='cpu').pin_memory("cuda")
        self.sqrt = torch.sqrt
        self.mean = torch.mean
        self.relu = torch.relu
        self.exp = torch.exp
        self.stack = torch.stack
        self.matvec = torch.mv

        # module def
        self.module = torch.nn.Module

        # pytorch function defs
        self.layerdef = lambda x: x

        # for everything in self, if its a tensor, send to cuda
        def sendToCuda(self, args, x):
            # create a new modifiable empty object
            class Empty:
                def __init__(self):
                    pass

            newself = Empty()
            for k, v in self.__dict__.items():
                if isinstance(v, torch.Tensor):
                    newself.__dict__[k] = v.cuda(non_blocking=True)

            ret = x(newself, *args)

            del newself
            return ret

        self.mainFunc = lambda x: x
        self.postfunc = lambda x: lambda self, *args: sendToCuda(self, args, x)
        self.layerdef = lambda x: lambda self, *args: sendToCuda(self, args, x)
        self.prefunc = lambda x: lambda *args: x(*args).cuda()
        self.emptyState = torch.zeros(
            4*layers, embed, dtype=torch.float32, device="cuda")+0.01


RwkvOpList: dict[str, type[RWKVOPS]] = {
    "tensorflow": RWKVTFOps,
    "pytorch": RWKVPTOps,
    "pytorch-compatible": RWKVPTCompatOps,
    "pytorch-cuda": RWKVCudaOps,
    "pytorch-stream": RWKVStreamOps
}
