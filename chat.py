########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import loadModel
import discord
from genericpath import exists
from typing import List
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

TEMPERATURE = 0.9
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

client = discord.Client(
    intents=discord.Intents.all())


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

# init empty save state and question context
init_state = model.empty_state()
model_tokens = tokenizer.tokenizer.encode(context)


saveStates = {}
saveStates["empty"] = ([187], init_state.clone())

# Put the prompt into the init_state
init_state = model.loadContext([], model_tokens, init_state)
saveStates["questions"] = (init_state[0], init_state[1].clone())

src_model_tokens = model_tokens.copy()
currstate = init_state


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
        new = f"User: {real_msg}\n\nRWKV:"
        tknew = tokenizer.tokenizer.encode(new)
        print(f'### add ###\n[{new}]')

        begin = len(currstate[0] + tknew)

        currstate = model.loadContext(currstate[0], tknew, currstate[1])
        for i in tqdm(range(100)):
            currstate = model.run(currstate[0], currstate[1])
            if (tokenizer.tokenizer.decode(currstate[0])[-2:] == '\n\n'):
                break
        send_msg = tokenizer.tokenizer.decode(currstate[0][begin:]).strip()
        print(f'### send ###\n[{send_msg}]')
        await message.reply(send_msg)

client.run(os.environ["TOKEN"])
