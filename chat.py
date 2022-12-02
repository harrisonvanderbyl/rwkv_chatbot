########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import loadModelForOnnx
import discord
from genericpath import exists
from typing import List
from torch.nn import functional as F
from src.model_run import RWKV_RNN
import numpy as np
import math
import os
import sys
import types
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
user = "User"
interface = ":"
bot = "RWKV"
context = f'''
The following is a conversation between a highly knowledgeable and intelligent AI assistant called {bot}, and a human user called {user}. In the following interactions, {user} and {bot} converse in natural language, and {bot} always answer {user}'s questions. {bot} is very smart, polite and humorous. {bot} knows a lot, and always tells the truth. The conversation begins.

{user}{interface} who is president of usa?

{bot}{interface} It’s Joe Biden; he was sworn in earlier this year.

{user}{interface} french revolution what year

{bot}{interface} It started in 1789, but it lasted 10 years until 1799.

{user}{interface} guess i marry who ?

{bot}{interface} Only if you tell me more about yourself - what are your interests?

{user}{interface} wat is lhc

{bot}{interface} It’s a large and very expensive piece of science equipment. If I understand correctly, it’s a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

'''
# context = "hello world! I am your supreme overlord!"
NUM_TRIALS = 999
LENGTH_PER_TRIAL = 200

TEMPERATURE = 0.9
top_p = 0.9
top_p_newline = 0.9  # only used in TOKEN_MODE = char

DEBUG_DEBUG = False  # True False --> show softmax output

########################################################################################################

pre, layers, post, emptyState = loadModelForOnnx.loadModel()


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

client = discord.Client(
    intents=discord.Intents.all())


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


def loadContext(self, ctx: list[int], newctx: list[int], statex):

    for i in tqdm(range(len(newctx))):

        x = ctx+newctx[:i+1]
        o = [[x[-1]], statex]
        ss = [pre]+self
        for s in ss:
            o = s.forward(*o)

    return ctx+newctx, o[1]


def sample_logits(ozut: torch.Tensor, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # turn to float if is half and cpu
    out = ozut
    probs = F.softmax(out, dim=-1)

    sorted_probs = torch.sort(probs, descending=True)[0]
    cumulative_probs = torch.cumsum(
        sorted_probs.float(), dim=-1).cpu().numpy()
    cutoff = float(sorted_probs[np.argmax(
        cumulative_probs > top_p_usual)])
    probs[probs < cutoff] = 0
    if temp != 1.0:
        probs = probs.pow(1.0 / temp)

    out: int = torch.multinomial(probs.float(), 1, True)[0]
    return out


# init empty save state and question context
init_state = emptyState.clone()
model_tokens = tokenizer.tokenizer.encode(context)


saveStates = {}
saveStates["empty"] = ([187, 187], init_state.clone())

# Put the prompt into the init_state
init_state = loadContext(layers, [187, 187], model_tokens, init_state)
saveStates["questions"] = (init_state[0], init_state[1].clone())

src_model_tokens = model_tokens.copy()
currstate = init_state

storys = []


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
        new = f"\n\{user}{interface} {real_msg}\n\n{bot}:"
        tknew = tokenizer.tokenizer.encode(new)
        print(f'### add ###\n[{new}]')

        begin = len(currstate[0] + tknew)

        currstate = loadContext(layers,
                                currstate[0], tknew, currstate[1])
        state = currstate
        with torch.no_grad():
            for i in tqdm(range(100)):
                ctx = state[0]

                state = (pre.preProcess[state[0][-1]], state[1])
                for r in layers:
                    state = r.forward(state[0], state[1])
                state = (ctx+[sample_logits(post.forward(state[0]))], state[1])
                if (state[0][-1] == 535 or (state[0][-2] == 187 and state[0][-1] == 187)):
                    break

        currstate = state
        send_msg = tokenizer.tokenizer.decode(currstate[0][begin:]).strip()
        print(f'### send ###\n[{send_msg}]')
        await message.reply(send_msg)
    if msg[:10] == '+drkv_gen ':

        real_msg = msg[10:].strip()
        new = f"{real_msg}".replace("\\n", "\n")
        tknew = tokenizer.tokenizer.encode(new)
        print(f'### add ###\n[{new}]')

        begin = len(tknew)

        state = loadContext(layers, [187, 187], tknew, emptyState.clone())

        with torch.no_grad():

            for i in tqdm(range(100)):

                ctx = state[0]

                state = (state[0][-1], state[1])
                lr = [pre] + layers + [post]
                for r in layers:
                    state = r.forward(*state)
                state = (ctx+[sample_logits(state[0])], state[1])

        send_msg = tokenizer.tokenizer.decode(state[0][begin:]).strip()
        print(f'### send ###\n[{send_msg}]')
        await message.reply(send_msg+"\n continue with +drkv_cont "+str(len(storys)))
        storys.append(state)

    if msg[:11] == '+drkv_cont ':
        real_msg = msg[11:].strip()

        state = storys[int(real_msg)]

        begin = len(state[0])

        with torch.no_grad():
            for i in tqdm(range(100)):
                ctx = state[0]

                state = (pre.preProcess[state[0][-1]], state[1])
                for r in layers:
                    state = r.forward(state[0], state[1])
                state = (ctx+[sample_logits(post.forward(state[0]))], state[1])

        send_msg = tokenizer.tokenizer.decode(state[0][begin:]).strip()
        print(f'### send ###\n[{send_msg}]')
        await message.reply(send_msg+"\n continue with +drkv_cont "+real_msg)
        storys[int(real_msg)] = state


client.run(os.environ["TOKEN"])
