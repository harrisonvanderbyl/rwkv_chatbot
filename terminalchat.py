########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import src.model_run_onnx as mro
import inquirer
import loadModelForOnnx
import os
import sys
import time
import gc
from scipy.special import softmax

import torch
from src.rwkvops import RwkvOpList
from src.utils import TOKENIZER
from tqdm import tqdm
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass

########################################################################################################
# Step 2: set prompt & sampling stuffs
########################################################################################################

# context = 'A'
# context = "\nIn the"
# context = '\nSugar:'

# context = "\n深圳是" # test Chinese
# context = "\n東京は" # test Japanese

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
top_p = 0.9
top_p_newline = 0.9  # only used in TOKEN_MODE = char

DEBUG_DEBUG = False  # True False --> show softmax output

########################################################################################################
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


# bot.py

# init empty save state and question context
model_tokens = tokenizer.tokenizer.encode(context)


def loadContext(ctx: list[int], statex, newctx: list[int]):
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    with torch.jit.optimized_execution(True):
        for i in tqdm(range(len(newctx))):

            x = ctx+newctx[:i+1]

            o = model.forward(x[:1], statex)
            statex = o[1]
            # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    return ctx+newctx, o[1]


print(tokenizer.tokenizer.encode("\n\n"))
#  see if save_state file exists

state = loadContext(newctx=model_tokens, ctx=[], statex=emptyState)


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


for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
    print("--")
    time_ref = time.time_ns()
    inp = input('User: ')
    if (inp.startswith("save ")):
        savestates[inp[5:]] = (state[0], state[1].clone())
        # Save to file
        torch.save(savestates, f"save_state.pt")
        print("Saved to file.")
        continue
    if (inp.startswith("load ")):

        savestates = torch.load(
            f"save_state.pt")
        state = (savestates[inp[5:]][0], savestates[inp[5:]][1].clone())

        continue
    state = loadContext(ctx=state[0], statex=state[1], newctx=tokenizer.tokenizer.encode(
        f"\n\nUser: {inp}\n\nRWKV:"))

    if TRIAL == 0:

        gc.collect()
        torch.cuda.empty_cache()

    with torch.no_grad():
        for i in range(100):
            ctx = state[0]
            state = model.forward(
                *state)
            mn = sample_logits(state[0], temp=1.0, top_p_usual=0.8)
            outchar = tokenizer.tokenizer.decode(mn)
            state = (ctx + [mn], state[1])
            if (outchar == "\n" or outchar == "\n\n"):
                break
            print(outchar, end="")
    state = state
