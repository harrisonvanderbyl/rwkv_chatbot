########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import loadModel
import os
import sys
import time
import gc
import torch
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

model = loadModel.loadModel()


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
print(tokenizer.tokenizer.encode("\n\n"))
#  see if save_state file exists
if os.path.isfile(f"save_states_{model.n_emb}_{model.n_layer}.pt"):
    print("Loading save state...")
    savestates = torch.load(f"save_states_{model.n_emb}_{model.n_layer}.pt")
    state = savestates["init"]

else:
    state = model.loadContext(newctx=model_tokens)
    savestates = {
        "init": (state[0], state[1].clone())
    }
    torch.save(savestates, f"save_states_{model.n_emb}_{model.n_layer}.pt")

for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
    print("--")
    time_ref = time.time_ns()
    inp = input('User: ')
    if (inp.startswith("save ")):
        savestates[inp[5:]] = (state[0], state[1].clone())
        # Save to file
        torch.save(savestates, f"save_states_{model.n_emb}_{model.n_layer}.pt")
        print("Saved to file.")
        continue
    if (inp.startswith("load ")):

        savestates = torch.load(
            f"save_states_{model.n_emb}_{model.n_layer}.pt")
        state = (savestates[inp[5:]][0], savestates[inp[5:]][1].clone())

        continue
    state = model.loadContext(ctx=state[0], statex=state[1], newctx=tokenizer.tokenizer.encode(
        f"\n\nUser: {inp}\n\nRWKV:"))

    state = [{"score": 1, "state": state[1], "ctx": state[0]}]

    if TRIAL == 0:

        gc.collect()
        torch.cuda.empty_cache()

    with torch.no_grad():
        for i in range(100):

            state = model.run(
                state, temp=TEMPERATURE, top_p=top_p)
            print(tokenizer.tokenizer.decode(state[0]["ctx"]))
    state = (state[0]["ctx"], state[0]["state"])
    print(tokenizer.tokenizer.decode(state[0]))
