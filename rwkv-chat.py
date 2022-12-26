import socketserver
import json
import http.server
from http import HTTPStatus
import webbrowser
import matplotlib.pyplot as plt
import loadModelForOnnx
import tqdm
import onnxruntime as ort
from genericpath import exists
from typing import List
import numpy as np
import os
import time
import gc
import torch
from src.utils import TOKENIZER
from src.rwkvops import RwkvOpList
import inquirer
from scipy.special import softmax
from torch.nn import functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import src.model_run_onnx as mro
from sty import Style, RgbFg, fg

# build website using yarn
# yarn build
os.chdir("web-interface")
os.system("yarn")
os.system("yarn build")
os.chdir("..")

fg.orange = Style(RgbFg(255, 150, 50))

# context = 'A'
# context = "\nIn the"
# context = '\nSugar:'
# context = "\n深圳是" # test Chinese
# context = "\n東京は" # test Japanese
# context = "\n深圳是" # test Chinese
# context = "\n東京は" # test Japanese

files = os.listdir()
# filter by ending in .pth
files = [f for f in files if f.endswith(".pth")]

questions = [
    inquirer.List('file',
                  message="What model do you want to use?",
                  choices=files,
                  ),
    inquirer.List(
        'method',
        message="What inference method?",
        choices=RwkvOpList.keys()
    )

]
q = inquirer.prompt(questions)

model, emptyState = mro.createRWKVModel(
    q["file"], mode=q["method"])

###### A good prompt for chatbot ######
context = '''
The following is a conversation between a highly knowledgeable and intelligent AI assistant, called RWKV, and a human user, called User. In the following interactions, User and RWKV will converse in natural language, and RWKV will do its best to answer User’s questions. RWKV was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. The conversation begins.

User: OK RWKV, I’m going to start by quizzing you with a few warm-up questions. Who is currently the president of the USA?

RWKV: It’s Joe Biden; he was sworn in earlier this year.

User: What year was the French Revolution?

RWKV: It started in 1789, but it lasted 10 years until 1799.

User: Can you guess who I might want to marry?

RWKV: Only if you tell me more about yourself - what are your interests?

User: Aha, I’m going to refrain from that for now. Now for a science question. What can you tell me about the Large Hadron Collider (LHC)?

RWKV: It’s a large and very expensive piece of science equipment. If I understand correctly, it’s a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.
'''
# context = "hello world! I am your supreme overlord!"
NUM_TRIALS = 999
LENGTH_PER_TRIAL = 200

TEMPERATURE = 1.0
top_p = 0.8
top_p_newline = 0.9  # only used in TOKEN_MODE = char

DEBUG_DEBUG = False  # True False --> show softmax output

########################################################################################################


print(f'\nOptimizing speed...')

gc.collect()
torch.cuda.empty_cache()

# input(0)

TOKEN_MODE = "pile"
WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None
print(f'\nLoading tokenizer {WORD_NAME}...')
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)
if TOKEN_MODE == "pile":
    assert tokenizer.tokenizer.decode([187]) == '\n'

########################################################################################################


ctx1 = tokenizer.tokenizer.encode(context)
src_ctx1 = ctx1.copy()


print(
    "Note: currently the first run takes a while if your prompt is long, as we are using RNN to preprocess the prompt. Use GPT to build the hidden state for better speed.\n"
)

time_slot = {}
time_ref = time.time_ns()


def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt


init_out = []

out = []

print("torch.cuda.memory_allocated: %fGB" %
      (torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB" %
      (torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB" %
      (torch.cuda.max_memory_reserved(0)/1024/1024/1024))


def loadContext(ctx: list[int], statex, newctx: list[int]):
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    with torch.jit.optimized_execution(True):
        for i in tqdm.tqdm(range(len(newctx))):

            x = ctx+newctx[:i+1]

            o = model.forward([x[-1]], statex)
            statex = o[1]
            # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    return ctx+newctx, o[1]


tokens = loadContext(ctx=[], newctx=ctx1, statex=emptyState)

origistate = (tokens[0], tokens[1])


def sample_logits(ozut, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
    ozut = ozut.numpy()
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # turn to float if is half and cpu
    probs = softmax(ozut, axis=-1)

    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temp != 1.0:
        probs = probs.pow(1.0 / temp)
    probs = probs / np.sum(probs, axis=0)
    mout = np.random.choice(a=len(probs), p=probs)

    return mout


newlinetok = tokenizer.tokenizer.encode('\n')[0]
double_newlinetok = tokenizer.tokenizer.encode('\n\n')[0]


class S(http.server.SimpleHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers",
                         "X-Requested-With, Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        self.send_response(200)
        if "css" in self.path:
            self.send_header('Content-type', 'text/css')
        elif "js" in self.path:
            self.send_header('Content-type', 'text/javascript')
        else:
            self.send_header('Content-type', 'text/html')

        self.end_headers()
        if (self.path == "/"):
            self.path = "/index.html"
        # self._set_response()
        self.wfile.write(
            open("/".join(__file__.split("/")[:-1])+"/web-interface/build/"+self.path, "rb").read())

    def do_POST(self):
        self.send_response(200)

        # Get body
        content_length = int(self.headers['Content-Length'])

        body = self.rfile.read(content_length)
        body = body.decode('utf-8')

        # get json
        body = json.loads(body)

        print(body)

        try:
            if (body["state"] is None):
                body["state"] = origistate[1]
            else:
                body["state"] = torch.tensor(body["state"]).to(
                    device=origistate[1].device, dtype=origistate[1].dtype)
        except:
            body["state"] = origistate[1]

        tokens = tokenizer.tokenizer.encode(
            "User:"+body["message"]+"\n\nRWKV:")

        currentData = loadContext(ctx=[], newctx=tokens, statex=body["state"])

        ln = len(currentData[0])

        for i in range(100):
            x, state = model.forward([currentData[0][-1]], currentData[1])
            token = sample_logits(x, temp=TEMPERATURE, top_p_usual=top_p)
            currentData = (currentData[0]+[token], state)
            if (token == newlinetok or token == double_newlinetok) and i > 0:
                break

        tokens = tokenizer.tokenizer.decode(currentData[0][ln:])

        # flatten
        print(tokens)

        out = {}

        out["message"] = tokens

        # recursively convert state to number from tensor
        state = currentData[1]
        for i in range(len(state)):
            state[i] = state[i].float().cpu().numpy().tolist()

        out["state"] = currentData[1]

        # set content length
        out = json.dumps(out).encode("utf8")
        self.send_header('Content-Length', len(out))
        self.send_header('Content-Type', 'text/json')
        self.send_header('Access-Control-Allow-Origin', '*')

        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write(out)


httpd = socketserver.TCPServer(('', int(input("Port:"))), S)
# open browser
if (input("Open browser? (y/N)") == "y"):
    webbrowser.open("http://localhost:"+str(httpd.server_address[1]))

httpd.serve_forever()
