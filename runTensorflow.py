from genericpath import exists
from typing import List
import numpy as np
import os
import time
import gc
import torch
from src.utils import TOKENIZER
import inquirer
from scipy.special import softmax
# context = 'A'
# context = "\nIn the"
# context = '\nSugar:'
import tqdm
# context = "\n深圳是" # test Chinese
# context = "\n東京は" # test Japanese
import tensorflow as tf

files = os.listdir("tf")


questions = [
    inquirer.List('file',
                  message="What model do you want to use?",
                  choices=files,
                  ),
]
loadFile = "tf/"+inquirer.prompt(questions)["file"]


embed = int(loadFile.split("-")[2])
layers = int(loadFile.split("-")[1])
floatmode = (loadFile.split("-")[3])
mm = (layers*5)*[embed*[0]]


if floatmode == "torch.float16":
    floatmode = torch.float16
elif floatmode == "torch.float32":
    floatmode = torch.float32
elif floatmode == "torch.bfloat16":
    floatmode = torch.bfloat16

# emptyState = torch.load(loadFile+"/emptyState.pt")


# pre = ort.InferenceSession(
#     f"{loadFile}/preprocess.onnx", providers=providers, sess_options=so)

options = inquirer.prompt([
    inquirer.List('type',
                  message="What model varient",
                  choices=["litefp32", "litefp16", "full"],
                  ),
])["type"]


class interOp:
    def __init__(self, sig) -> None:
        self.sig = sig
        # print(tf.lite.experimental.load_delegate("delegate.so"))
        if (options == "litefp32"):
            self.model = tf.lite.Interpreter(
                model_path=loadFile+f"/{sig}/model_float32.tflite")
        elif (options == "litefp16"):
            self.model = tf.lite.Interpreter(
                model_path=loadFile+f"/{sig}/model_float16.tflite")
        elif (options == "full"):
            self.model = tf.saved_model.load(
                loadFile+f"/{sig}")

    def run(self, *x):
        if (options != "full"):
            self.model.allocate_tensors()

            for i, inp in enumerate(x):
                self.model.set_tensor(
                    self.model.get_input_details()[i]["index"], inp)

            self.model.invoke()
            outs = self.model.get_output_details()
            # print(self.sig, len(outs), [outs[o]["shape"]
            #       for o in range(len(outs))])
            return [self.model.get_tensor(out["index"]) for out in outs]
        else:
            rx: tf.Module = self.model
            if (len(x) == 1):
                out = rx(x[0])
                return [out]
            elif (len(x) == 2):
                out = rx([*x])
                return out


# my_signature is callable with input as arguments.
pre = interOp("pre")
post = interOp("post")

layernames = os.listdir(loadFile)
layernames = [l for l in layernames if l.startswith("layer")]
layernames.sort()
layers = []
for l in layernames:
    layers += [interOp(l)]


# prea = pre.run(tf.Variable([192], dtype=tf.int32))
# print(prea[0])


emptyState = tf.Variable(mm, dtype=tf.float32)


# layers = os.listdir(loadFile)
# layers = filter(lambda x: "layer" in x, layers)
# layers = list(layers)
# layers.sort()
# print(layers)
# layers = list(map(lambda x: ort.InferenceSession(
#     f"{loadFile}/{x[1]}", providers=providers2[x[0]], sess_options=so), enumerate(layers)))
# ###### A good prompt for chatbot ######
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
    statex = statex.numpy()
    for i in tqdm.tqdm(range(len(newctx))):
        x = ctx+newctx[:i+1]
        x, = pre.run(tf.Variable([x[-1]], tf.int32))

        for l in layers:
            statex, x = l.run(x, statex)

        # print(o[0][0] - statex[0][0])
        # print(statex[2][5] - lmx)

    return ctx+newctx, statex


tokens = loadContext(ctx=[], newctx=ctx1, statex=emptyState)


def sample_logits(ozut: torch.Tensor, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
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
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)

    return out


for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
    print("--")
    time_ref = time.time_ns()

    if TRIAL == 0:

        gc.collect()
        torch.cuda.empty_cache()

    record_time('preprocess')
    with torch.no_grad():
        for i in range(100):
            chars: List[int] = tokens[0]

            state = tokens[1]

            x, = pre.run(tf.Variable([chars[-1]], dtype=tf.int32))

            for l in layers:
                state, x = l.run(x, state)
            myout = (post.run(x), state)

            chars += [sample_logits(
                myout[0][0], temp=TEMPERATURE, top_p_usual=top_p)]
            char = tokenizer.tokenizer.decode(chars[-1])

            tokens = (chars, myout[1])

            if '\ufffd' not in char:
                print(char, end="", flush=True)

    record_time('total')
    # print(f'\n\n{time_slot}\n\n')
    print(
        f"\n\n--- preprocess {round(time_slot['preprocess'], 2)}s, generation {round(time_slot['total']-time_slot['preprocess'], 2)}s ", end=''
    )

print(("-" * 50) + '\n')
