import discord
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


def loadContext(ctx: list[int], newctx: list[int], statex):
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


# bot.py


client = discord.Client(
    intents=discord.Intents.all())


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


# init empty save state and question context
init_state = emptyState
model_tokens = tokenizer.tokenizer.encode(context)


saveStates = {}
saveStates["empty"] = ([187, 187], init_state.clone())

# Put the prompt into the init_state
init_state = loadContext([187, 187], model_tokens, init_state)
saveStates["questions"] = (init_state[0], init_state[1])

src_model_tokens = model_tokens.copy()
currstate = init_state

storys = []


def s0(probs, temp: float = 1.0, top_p_usual: float = 0.8) -> int:

    sorted_probs = torch.sort(probs, descending=True)[0]
    cumulative_probs = torch.cumsum(
        sorted_probs.float(), dim=-1).cpu().numpy()
    cutoff = float(sorted_probs[np.argmax(
        cumulative_probs > top_p_usual)])
    probs[probs < cutoff] = 0
    if temp != 1.0:
        probs = probs.pow(1.0 / temp)

    out = torch.multinomial(probs.float(), 1, True)[0]
    # print(sorted_probs[:3])
    return out


def s1(ozut, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
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


sample_logits = [s0, s1][int(input("sample method: 0 for torch, 1 for numpy"))]


@client.event
async def on_message(message):
    global model_tokens, currstate
    # print(
    #     f"message received({message.guild.name}:{message.channel.name}):", message.content)

    if message.author.bot:
        return

    msg = message.content.strip()

    if msg == '+reset_drkv' or msg == '+drkv_reset':
        model_tokens = tokenizer.tokenizer.encode(context)
        currstate = init_state

        await message.reply(f"Chat reset. This is powered by RWKV-4 Language Model.")
        return

    if msg[:11] == '+drkv_save ':
        saveStates[msg[11:]] = (currstate[0], currstate[1].clone())
        await message.reply(f"Saved state {msg[11:]}")
        return

    if msg[:11] == '+drkv_load ':
        if msg[11:] in saveStates:
            currstate = saveStates[msg[11:]]
            await message.reply(f"Loaded state {msg[11:]}")
        else:
            await message.reply(f"State {msg[11:]} not found")
        return

    if msg[:11] == '+drkv_list ':
        await message.reply(f"Saved states: {', '.join(saveStates.keys())}")
        return
    if msg[:6] == '+drkv ':

        real_msg = msg[6:].strip()
        new = f"\nUser: {real_msg}\n\nRWKV:"
        tknew = tokenizer.tokenizer.encode(new)
        print(f'### add ###\n[{new}]')

        currstate = loadContext(currstate[0], tknew, currstate[1])
        begin = len(currstate[0])
        state = currstate

        with torch.no_grad():
            for i in range(100):
                o = model.forward([state[0][-1]], state[1])
                tok = sample_logits(o[0])
                print(tokenizer.tokenizer.decode(tok), end='')
                state = (state[0] + [tok], o[1])
                if (tok == 187):
                    break

        send_msg = tokenizer.tokenizer.decode(state[0][begin:]).strip()
        currstate = state
        if (len(send_msg) == 0):
            send_msg = "Error: No response generated."
        print(f'### send ###\n[{send_msg}]')
        await message.reply(send_msg)
    if msg[:10] == '+drkv_gen ':

        real_msg = msg[10:].strip()
        new = f"{real_msg}".replace("\\n", "\n")
        tknew = tokenizer.tokenizer.encode(new)
        print(f'### add ###\n[{new}]')

        begin = len(tknew)

        state = loadContext([187, 187], tknew, emptyState)

        with torch.no_grad():
            for i in range(100):
                o = model.forward([state[0][-1]], state[1])
                tok = sample_logits(o[0])
                state = (state[0] + [tok], o[1])

        send_msg = tokenizer.tokenizer.decode(state[0][begin:]).strip()
        state = state
        print(f'### send ###\n[{send_msg}]')
        await message.reply(send_msg+"\n continue with +drkv_cont "+str(len(storys)))
        storys.append(state)

    if msg[:11] == '+drkv_cont ':
        real_msg = msg[11:].strip()

        state = storys[int(real_msg)]

        begin = len(state[0])

        with torch.no_grad():
            for i in range(100):
                o = model.forward([state[0][-1]], state[1])
                tok = sample_logits(o[0])
                state = (state[0] + [tok], o[1])

        send_msg = tokenizer.tokenizer.decode(state[0][begin:]).strip()
        print(f'### send ###\n[{send_msg}]')
        await message.reply(send_msg+"\n continue with +drkv_cont "+real_msg)
        storys[int(real_msg)] = state


client.run(os.environ["TOKEN"])
