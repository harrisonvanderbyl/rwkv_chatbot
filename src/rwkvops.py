import os
import tensorflow as tf
import torch
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import http.server
import json
import logging
import socketserver


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
        self.initfunc: notimplemented
        self.layerdef: notimplemented
        self.mainfunc: notimplemented
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
        self.initfunc = lambda x: x
        self.layerdef = tf.function(
            input_signature=5*[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        self.mainfunc = tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.int32), tf.TensorSpec(
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
        self.stack = lambda x: x
        self.matvec = torch.mv

        # module def
        self.module = torch.nn.Module

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x
        self.postfunc = lambda x: x
        self.prefunc = lambda x: x
        self.emptyState = torch.zeros(
            4*layers, embed, dtype=torch.float32)+0.01


class RWKVPTCompatOps(RWKVPTOps):
    def __init__(self, layers, embed):
        super().__init__(layers, embed)
        self.relu = lambda x: torch.max(x, torch.zeros_like(x))
        self.matvec = lambda x, y: torch.sum(x*y, dim=1)


class RWKVCudaOps(RWKVPTOps):
    def __init__(self, layers, embed):
        super().__init__(layers, embed)
        self.initTensor = lambda x: torch.tensor(x, device='cuda')
        self.emptyState = torch.zeros(
            4*layers, embed, dtype=torch.float32, device="cuda")+0.01


class RWKVExportOnnxOps(RWKVPTOps):
    def __init__(self, layers, embed):
        path = f"onnx/rwkv-{layers}-{embed}-{torch.float32}/"
        super().__init__(layers, embed)
        self.stack = torch.stack

        def export(self, x, state):
            print("exporting")
            try:
                try:
                    os.mkdir("onnx")
                except:
                    pass
                os.mkdir(path)
            except:
                pass
            torch.onnx.export(
                self.preprocess, (torch.zeros(1, dtype=torch.int32),), f"{path}pre.onnx")
            torch.onnx.export(
                self.postprocess, (torch.zeros(embed, dtype=torch.float32),), f"{path}post.onnx")
            for i, layer in enumerate(self.mylayers):
                torch.onnx.export(
                    layer, (torch.zeros(embed, dtype=torch.float32)+0.01, torch.zeros(embed, dtype=torch.float32)+0.01, torch.zeros(embed, dtype=torch.float32)+0.01, torch.zeros(embed, dtype=torch.float32)+0.01, torch.zeros(embed, dtype=torch.float32)+0.01), f"{path}{i}.onnx")

            exit()
        self.mainfunc = lambda x: export


# class RWKVP2POps(RWKVCudaOps):
#     def __init__(self, layers, embed):
#         super().__init__(layers, embed)

#         def intFunc(self, pre, post, layers, x):
#             self.start = int(input(f"StartLayer(0-{len(layers)}):"))
#             self.end = min(
#                 int(input(f"EndLayer({self.start}-{len(layers)}):")), len(layers))

#             x(self, pre, post, layers[self.start:self.end])

#         def forward(self, x, state):
#             print("forward test")
#             return x

#         self.initfunc = lambda x: lambda self, pre, post, layers: intFunc(
#             self, pre, post, layers, x)

#         self.mainfunc = lambda rx: lambda self, x, state: forward(
#             self, x, state)


# class RWKVP2PServerOps(RWKVCudaOps):
#     def __init__(self, layers, embed):
#         super().__init__(layers, embed)

#         class S(http.server.SimpleHTTPRequestHandler):
#             def do_GET(self):
#                 self.send_response(200)
#                 self.send_header('Content-type', 'text/html')
#                 self.end_headers()
#                 # self._set_response()
#                 print(self.path)
#                 if (self.path.startswith("/files")):
#                     file = self.path.split("/")[2]

#                     self.wfile.write(
#                         open("/".join(__file__.split("/")[:-1])+"/onnxServer/"+file, "rb").read())
#                 else:
#                     self.wfile.write("RWKV SERVER".encode('utf-8'))

#             def do_POST(self):
#                 self.send_response(200)
#                 # Get body
#                 content_length = int(self.headers['Content-Length'])
#                 body = self.rfile.read(content_length)
#                 body = body.decode('utf-8')
#                 body = body.strip()

#                 print(body)

#                 tokens = tokenizer.encode(body)

#                 tokens = [preProcess.run(None, createInput(
#                     preProcess.get_inputs(), [[x], emptyState]))[0].tolist() for x in tokens]

#                 # flatten
#                 print(tokens)

#                 # convert to json
#                 tokens = json.dumps(tokens).encode("utf8")

#                 # set content length
#                 out = tokens
#                 self.send_header('Content-Length', len(out))
#                 self.send_header('Content-Type', 'text/json')

#                 self.send_response(HTTPStatus.OK)
#                 self.end_headers()
#                 self.wfile.write(out)

#             def do_PUT(self):
#                 self.send_response(200)
#                 # Get body
#                 content_length = int(self.headers['Content-Length'])
#                 body = self.rfile.read(content_length)
#                 body = body.decode('utf-8')
#                 body = json.loads(body)

#                 # array is a list of integers like "1,2,3,4" turn into array
#                 print(body)

#                 tokens = tokenizer.decode(body)

#                 self.send_response(HTTPStatus.OK)

#                 out = tokens.encode('utf-8')

#                 # set content length
#                 self.send_header('Content-Length', len(out))
#                 self.send_header('Content-Type', 'text/json')

#                 self.end_headers()

#                 print(out)
#                 self.wfile.write(out)

#         httpd = socketserver.TCPServer(('', 8088), S)

#         def intFunc(self, pre, post, layers, x):
#             x(self, pre, post, [])

#         def forward(self, x, state):

#             httpd.serve_forever()
#             print("forward test")
#             return x

#         self.initfunc = lambda x: lambda self, pre, post, layers: intFunc(
#             self, pre, post, layers, x)

#         self.mainfunc = lambda rx: lambda self, x, state: forward(
#             self, x, state)


class RWKVStreamOps(RWKVPTOps):
    def __init__(self, layers, embed):
        super().__init__(layers, embed)
        self.initTensor = lambda x: torch.tensor(
            x, device='cpu').pin_memory("cuda")

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

        self.postfunc = lambda x: lambda self, *args: sendToCuda(self, args, x)
        self.layerdef = lambda x: lambda self, *args: sendToCuda(self, args, x)
        self.prefunc = lambda x: lambda *args: x(*args).cuda()
        self.emptyState = torch.zeros(
            4*layers, embed, dtype=torch.float32, device="cuda")+0.01


class RWKVStreamBigOps(RWKVPTOps):
    def __init__(self, layers, embed):
        super().__init__(layers, embed)

        processDtype = torch.float64
        storageDtype = torch.bfloat16

        target = float(
            input("Designate the amount of memory to allocate (in GB):"))
        self.initTensor = lambda x: torch.tensor(
            x, device='cpu', dtype=storageDtype if len(x.shape) == 2 else processDtype).pin_memory("cuda") if (torch.cuda.max_memory_reserved(0)/1024/1024/1024) > target else torch.tensor(x, dtype=storageDtype if len(x.shape) == 2 else processDtype).cuda()

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

        self.postfunc = lambda x: lambda self, * \
            args: sendToCuda(self, args, x).float()
        self.layerdef = lambda x: lambda self, *args: sendToCuda(self, args, x)
        self.prefunc = lambda x: lambda *args: x(*args).cuda()
        self.matvec = lambda z, y: z.mv(y.to(storageDtype)).to(processDtype)
        self.emptyState = torch.zeros(
            4*layers, embed, dtype=processDtype, device="cuda")+0.01


RwkvOpList: dict[str, type[RWKVOPS]] = {
    "tensorflow": RWKVTFOps,
    "pytorch": RWKVPTOps,
    "pytorch-compatible": RWKVPTCompatOps,
    "pytorch-cuda": RWKVCudaOps,
    "pytorch-stream": RWKVStreamOps,
    "pytorch-stream-target": RWKVStreamBigOps,
    "export-onnx": RWKVExportOnnxOps,

    # "pytorch-p2p": RWKVP2POps,
}
