import socketserver
import json
import http.server
from http import HTTPStatus
import webbrowser
from rwkvChatPersonalities import Personaities
import tqdm
import numpy as np
import os
import time
from typing import Dict as dict
from typing import List as list
import gc


from src.utils import TOKENIZER

try:
    import inquirer
except:
    install = not input(
        "inquirer not installed. Would you like to install it? (Y/n)") in ["n", "N"]
    if install:
        os.system("pip3 install inquirer")
        import inquirer
    else:
        print("Exiting...")
        exit()
try:
    from scipy.special import softmax
except:
    install = inquirer.prompt([inquirer.Confirm(
        'install',
        message="scipy not installed. Would you like to install it?",
        default=True,
    )])["install"]
    if install:
        os.system("pip3 install scipy")
        from scipy.special import softmax
    else:
        print("Exiting...")
        exit()

try:
    import torch
except:
    install = inquirer.prompt([inquirer.Confirm(
        'install',
        message="torch not installed. Would you like to install it? NOTE: AMD GPU users should double check https://pytorch.org/get-started/locally/",
        default=True,
    )])["install"]
    if install:
        # test if amd gpu

        os.system(
            "pip3 install torch torchvision torchaudio")
        import torch
    else:
        print("Exiting...")
        exit()
from src.rwkv import RWKV, Backends


# check if yarn installed
if os.system("yarn --version"):
    install = inquirer.prompt([inquirer.Confirm(
        'install',
        message="yarn not installed. Would you like to install it?",
        default=True,
    )])["install"]
    if install:
        if os.system("npm install -g yarn"):
            print("Failed to install yarn using npm. you need to install nodejs manually from https://nodejs.org/en/download/")
            exit()
    else:
        print("Exiting...")
        exit()
# build website using yarn
# yarn build
os.chdir("web-interface")
os.system("yarn")
os.system("yarn build")
os.chdir("..")


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
DownloadPrompt = "Download more models..."
questions = [
    inquirer.List('file',
                  message="What model do you want to use?",
                  choices=files+[DownloadPrompt],
                  ),


]

downloadLinks = [{
    "name": "RWKV-14B: recomended specs: 32GB VRAM, 2x4060FE",
    "link": "https://huggingface.co/BlinkDL/rwkv-4-pile-14b/resolve/main/RWKV-4-Pile-14B-20221217-3794.pth",
},
    {
    "name": "RWKV-7B: recomended specs: 16GB VRAM, 4060FE/2x2080",
    "link": "https://huggingface.co/BlinkDL/rwkv-4-pile-7b/resolve/main/RWKV-4-Pile-7B-20221123-ctx2048.pth"
},
    {
    "name": "RWKV-3B: recomended specs: 8GB VRAM, 2080/2x1060-6G",
    "link": "https://huggingface.co/BlinkDL/rwkv-4-pile-3b/resolve/main/RWKV-4-Pile-3B-20221110-ctx4096.pth"
},
]

q = inquirer.prompt(questions)

if q["file"] == DownloadPrompt:
    print(f"\n NOTE: The following models can, and do, run on lower spec machines, through the user of layer streaming. You can download smaller models https://huggingface.co/BlinkDL here, however they are not recommended for use on this chatbot. \n")
    toDownload = inquirer.prompt([inquirer.List('model',
                                                message="What model do you want to download?",
                                                choices=[d["name"]
                                                         for d in downloadLinks],
                                                )])
    os.system(
        f"wget {downloadLinks[[d['name'] for d in downloadLinks].index(toDownload['model'])]['link']}")

method = inquirer.prompt([inquirer.List(
    'method',
    message="What inference method?",
    choices=Backends.keys())]
)

model, emptyState = RWKV(
    q["file"], mode=method["method"])

###### A good prompt for chatbot ######

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


people: dict[str, str] = {}


def sample_logits(ozut, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
    try:
        ozut = ozut.numpy()
    except:
        pass
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # turn to float if is half and cpu
    probs = softmax(ozut, axis=-1)

    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p_usual)])
    probs[probs < cutoff] = 0
    if temp != 1.0:
        probs = probs.pow(1.0 / temp)
    probs = probs / np.sum(probs, axis=0)
    mout = np.random.choice(a=len(probs), p=probs)

    return mout


newlinetok = tokenizer.tokenizer.encode('\n')[0]
double_newlinetok = tokenizer.tokenizer.encode('\n\n')[0]

progress = {}


def loadContext(ctx: list[int], statex, newctx: list[int], id="none"):
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    with torch.jit.optimized_execution(True):
        for i in tqdm.tqdm(range(len(newctx))):

            x = ctx+newctx[:i+1]
            progress[id] = x

            o = model.forward([x[-1]], statex)
            statex = o[1]
            # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    return ctx+newctx, o[1]


for P, personality in Personaities.items():
    people[P] = loadContext(ctx=[], newctx=tokenizer.tokenizer.encode(
        personality), statex=emptyState)[1]


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
        if (self.path == "/personalities.json"):
            self.wfile.write(json.dumps(list(people.keys())).encode('utf-8'))
            return
        if ("/progress" in self.path):
            self.wfile.write(
                tokenizer.tokenizer.decode(progress.get(self.path.split("/")[-1], tokenizer.tokenizer.encode("..."))).encode('utf-8'))
            return
        # self._set_response()
        self.wfile.write(
            open("web-interface/build/"+self.path, "rb").read())

    def do_POST(self):
        self.send_response(200)

        # Get body
        content_length = int(self.headers['Content-Length'])

        body = self.rfile.read(content_length)
        body = body.decode('utf-8')

        # get json
        body = json.loads(body)
        print(body["message"])

        body["state"] = body.get("state", None)
        character = body.get("character", list(people.keys())[0])

        if body["state"] is None:
            body["state"] = people[character]
        else:
            body["state"] = model.ops.initTensor(torch.tensor(body["state"]))

        progresskey = body["key"]

        tokens = tokenizer.tokenizer.encode(
            "\nUser:"+body["message"].replace("User:", ""), ""+f"END\n{character}:")

        currentData = loadContext(
            ctx=[], newctx=tokens, statex=body["state"], id=progresskey)

        ln = len(currentData[0])

        for i in range(400):
            progress[progresskey] = currentData[0]
            x, state = model.forward([currentData[0][-1]], currentData[1])
            if isinstance(x, torch.Tensor):
                x = x.cpu().float()
            token = sample_logits(x, temp=TEMPERATURE, top_p_usual=top_p)
            currentData = (currentData[0]+[token], state)
            if tokenizer.tokenizer.decode(currentData[0]).strip().endswith(("END", "User:", "\nEnd")):
                break

        tokens = tokenizer.tokenizer.decode(currentData[0][ln:])

        # flatten
        print(tokens)

        out = {}

        out["message"] = tokens

        # recursively convert state to number from tensor
        state = currentData[1]
        ns = []
        for i in range(len(state)):
            try:
                ns += [np.array(state[i].float().cpu()).tolist()]
            except:
                ns += [np.array(state[i]).tolist()]

        out["state"] = ns

        # set content length
        out = json.dumps(out).encode("utf8")
        self.send_header('Content-Length', len(out))
        self.send_header('Content-Type', 'text/json')
        self.send_header('Access-Control-Allow-Origin', '*')

        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write(out)


httpd = socketserver.ThreadingTCPServer(('', int(input("Port:"))), S)
# open browser
if (input("Open browser? (y/N)") == "y"):
    webbrowser.open("http://localhost:"+str(httpd.server_address[1]))

httpd.serve_forever()
