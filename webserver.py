from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import http.server
import json
import logging
import socketserver
import onnxruntime as ort
from transformers import PreTrainedTokenizerFast
emptyState = (4)*[12*[768*[0.01]]]
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="20B_tokenizer.json")

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.log_severity_level = 3
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]

providers2 = [providers]*3 + 10*[[
    ('CUDAExecutionProvider', {
        'device_id': 1,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]]
preProcess = ort.InferenceSession(
    f"./onnxServer/preprocess.onnx", providers=providers, sess_options=so)


def createInput(inputNames, values):
    inputs = {}
    for i, name in enumerate(inputNames):
        inputs[name.name] = values[i]
    return inputs


class S(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        # self._set_response()
        print(self.path)
        if (self.path.startswith("/files")):
            file = self.path.split("/")[2]

            self.wfile.write(
                open("/".join(__file__.split("/")[:-1])+"/onnxServer/"+file, "rb").read())
        else:
            self.wfile.write("RWKV SERVER".encode('utf-8'))

    def do_POST(self):
        self.send_response(200)
        # Get body
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        body = body.decode('utf-8')
        body = body.strip()

        print(body)

        tokens = tokenizer.encode(body)

        tokens = [preProcess.run(None, createInput(
            preProcess.get_inputs(), [[x], emptyState]))[0].tolist() for x in tokens]

        # flatten
        print(tokens)

        # convert to json
        tokens = json.dumps(tokens).encode("utf8")

        # set content length
        out = tokens
        self.send_header('Content-Length', len(out))
        self.send_header('Content-Type', 'text/json')

        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write(out)

    def do_PUT(self):
        self.send_response(200)
        # Get body
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        body = body.decode('utf-8')
        body = json.loads(body)

        # array is a list of integers like "1,2,3,4" turn into array
        print(body)

        tokens = tokenizer.decode(body)

        self.send_response(HTTPStatus.OK)

        out = tokens.encode('utf-8')

        # set content length
        self.send_header('Content-Length', len(out))
        self.send_header('Content-Type', 'text/json')

        self.end_headers()

        print(out)
        self.wfile.write(out)


httpd = socketserver.TCPServer(('', 8087), S)
httpd.serve_forever()
