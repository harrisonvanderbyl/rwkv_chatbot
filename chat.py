########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import discord
from src.model_run import RWKV_RNN
import numpy as np
import os
import copy
import types
import torch
from src.utils import TOKENIZER
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model

UNKNOWN_CHAR = None
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)

MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20220928-5147'
n_layer = 32
n_embd = 2560

ctx_len = 1024
os.environ["RWKV_FLOAT_MODE"] = "fp32"  # currently only supprts fp32
os.environ["RWKV_RUN_DEVICE"] = "cuda"  # 'cpu' (already very fast) or 'cuda'
model_type = "RWKV"  # 'RWKV' or 'RWKV-ffnPre'
user = "User"
bot = "Bot"
interface = ":"

init_prompt = f'''
The following is a conversation between a highly knowledgeable and intelligent AI assistant called {bot}, and a human user called {user}. In the following interactions, {user} and {bot} will converse in natural language, and {bot} will do its best to answer {user}'s questions. {bot} is respectful, polite and inclusive. {bot} knows a lot, and always tells the truth. The conversation begins.

{user}{interface} who is president of us?

{bot}{interface} It’s Joe Biden; he was sworn in earlier this year.

{user}{interface} what year is french revolution

{bot}{interface} It started in 1789, but it lasted 10 years until 1799.

{user}{interface} guess who i will marry ?

{bot}{interface} Only if you tell me more about yourself - what are your interests?

{user}{interface} what is lhc

{bot}{interface} It’s a large and very expensive piece of science equipment. If I understand correctly, it’s a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

'''

TEMPERATURE = 1.5
top_p = 0.8

# Load Model

print('loading...')
model = RWKV_RNN(
    MODEL_NAME, os.environ["RWKV_RUN_DEVICE"], model_type, n_layer, n_embd, ctx_len
)
model_tokens = []

########################################################################################################


def run_rnn(tokens, newline_adj=0):
    global model_tokens
    for i in range(len(tokens)):
        model_tokens += [tokens[i]]
        if i == len(tokens) - 1:
            out = model.forward(model_tokens)
        else:
            model.forward(model_tokens, preprocess_only=True)

    # print(f'### model ###\n[{tokenizer.tokenizer.decode(model_tokens)}]')

    out[0] = -999999999  # disable <|endoftext|>
    out[187] += newline_adj
    return out


all_state = {}


def save_all_stat(name, last_out):
    all_state[name] = {}
    all_state[name]['out'] = last_out
    all_state[name]['rnn'] = types.SimpleNamespace()
    model.save(all_state[name]['rnn'])
    all_state[name]['token'] = copy.deepcopy(model_tokens)


def load_all_stat(name):
    global model_tokens
    model.load(all_state[name]['rnn'])
    model_tokens = copy.deepcopy(all_state[name]['token'])
    return all_state[name]['out']

########################################################################################################

# Run inference


model.clear()
out = run_rnn(tokenizer.tokenizer.encode(init_prompt))
save_all_stat('chat_init', out)
save_all_stat('chat', out)

print(f'### prompt ###\n[{tokenizer.tokenizer.decode(model_tokens)}]\n')

# bot.py

client = discord.Client(
    intents=discord.Intents.all())


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


@client.event
async def on_message(message):
    global model_tokens
    # print(
    #     f"message received({message.guild.name}:{message.channel.name}):", message.content)

    if message.author.bot:
        return

    msg = message.content.strip()

    if msg == '+reset_rwkv' or msg == '+drkv_reset':
        out = load_all_stat('chat_init')
        save_all_stat('chat', out)
        await message.reply("Chat reset. This is powered by RWKV-4 3B Language Model.")
        return

    elif msg[:10] == '+drkv_gen ' or msg[:9] == '+drkv_qa ' or msg == '+drkv_more' or msg == '+drkv_retry' or msg == '+drkv_again':
        if msg[:10] == '+drkv_gen ':
            new = '\n' + msg[10:].strip()
            print(f'### prompt ###\n[{new}]')
            model.clear()
            out = run_rnn(tokenizer.tokenizer.encode(new))
            save_all_stat('gen_0', out)
        elif msg[:9] == '+drkv_qa ':
            new = f'\nQ: {msg[9:].strip()}\nA:'
            print(f'### prompt ###\n[{new}]')
            model.clear()
            out = run_rnn(tokenizer.tokenizer.encode(new))
            save_all_stat('gen_0', out)
        elif msg == '+drkv_more':
            try:
                out = load_all_stat('gen_1')
                save_all_stat('gen_0', out)
            except:
                return
        elif msg == '+drkv_retry' or msg == '+drkv_again':
            try:
                out = load_all_stat('gen_0')
            except:
                return
        begin = len(model_tokens)
        for i in range(100):
            token = tokenizer.sample_logits(
                out,
                model_tokens,
                ctx_len,
                temperature=TEMPERATURE,
                top_p_usual=top_p,
                top_p_newline=top_p,
            )
            out = run_rnn([token])
        send_msg = tokenizer.tokenizer.decode(model_tokens[begin:]).strip()
        print(f'### send ###\n[{send_msg}]')
        await message.reply(send_msg)
        save_all_stat('gen_1', out)

    elif msg[:6] == '+drkv ':
        out = load_all_stat('chat')

        real_msg = msg[6:].strip()
        new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
        print(f'### add ###\n[{new}]')

        out = run_rnn(tokenizer.tokenizer.encode(new), -999999999)
        begin = len(model_tokens)
        for i in range(100):
            if i <= 0:
                newline_adj = -999999999
            elif i <= 30:
                newline_adj = -2
            elif i <= 70:
                newline_adj = 0
            elif i <= 97:
                newline_adj = i - 70
            else:
                newline_adj = 999999999
            token = tokenizer.sample_logits(
                out,
                model_tokens,
                ctx_len,
                temperature=TEMPERATURE,
                top_p_usual=top_p,
                top_p_newline=top_p,
            )
            out = run_rnn([token], newline_adj)
            if tokenizer.tokenizer.decode(model_tokens[-10:]).endswith(f'\n\n'):
                break

        send_msg = tokenizer.tokenizer.decode(model_tokens[begin:]).strip()
        print(f'### send ###\n[{send_msg}]')
        await message.reply(send_msg)
        save_all_stat('chat', out)

client.run(os.environ['TOKEN'])
