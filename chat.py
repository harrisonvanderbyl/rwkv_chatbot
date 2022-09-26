########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from src.utils import TOKENIZER
import torch
import numpy as np
from src.model_run import RWKV_RNN
import os
import discord
import subprocess
print("start")
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# Settings

# Load Tokenizer

WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model


UNKNOWN_CHAR = None
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)

MODEL_NAME = '/fsx/BlinkDL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20220925-4537'  # filename for 1b5
n_layer = 32  # 24:1b5 32:3b
n_embd = 2560  # 2048 1b5 2560 3B

ctx_len = 1024
os.environ["RWKV_FLOAT_MODE"] = "fp32"  # currently only supprts fp32
os.environ["RWKV_RUN_DEVICE"] = "cuda"  # 'cpu' (already very fast) or 'cuda'
model_type = "RWKV"  # 'RWKV' or 'RWKV-ffnPre'
name = "User"
endmessage = "\n\n"
bot = "RWKV"
interface = ":"

# The following is a conversation between a highly knowledgeable and intelligent AI assistant, called RWKV, and a human user, called User. RWKV will do its best to answer User’s questions. RWKV knows a lot, and always tells the truth. The conversation begins.

context = f'''
The following is a conversation between a highly knowledgeable and intelligent AI assistant, called RWKV, and a human user, called User. In the following interactions, User and RWKV will converse in natural language, and RWKV will do its best to answer User’s questions. RWKV was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. The conversation begins.

{name}{interface} OK RWKV, I’m going to start by quizzing you with a few warm-up questions. Who is currently the president of the USA?

{bot}{interface} It’s Joe Biden; he was sworn in earlier this year.

{name}{interface} What year was the French Revolution?

{bot}{interface} It started in 1789, but it lasted 10 years until 1799.

{name}{interface} Can you guess who I might want to marry?

{bot}{interface} Only if you tell me more about yourself - what are your interests?

{name}{interface} Aha, I’m going to refrain from that for now. Now for a science question. What can you tell me about the Large Hadron Collider (LHC)?

{bot}{interface} It’s a large and very expensive piece of science equipment. If I understand correctly, it’s a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.'''

TEMPERATURE = 1.0
top_p = 0.8

# Load Model

print('loading...')
model = RWKV_RNN(
    MODEL_NAME, os.environ["RWKV_RUN_DEVICE"], model_type, n_layer, n_embd, ctx_len
)


########################################################################################################


def insert_data(model, tokens, ctx, clear=False, preprocess=True):  # T<abc> => model

    if (clear):
        model.clear()

    for i in range(len(ctx)):
        x = tokens + ctx[: i + 1]
        model.forward(x, preprocess_only=preprocess)

    return tokens + ctx


def predict_next(ctx):  # T<abc> => T<abcd>
    x = ctx
    x = x[-ctx_len:]
    out = model.forward(x)

    out[0] = -999999999  # disable <|endoftext|>
    out[187] -= 1

    char = tokenizer.sample_logits(
        out,
        x,
        ctx_len,
        temperature=TEMPERATURE,
        top_p_usual=top_p,
        top_p_newline=0.9,
    )
    return ctx + [char]


########################################################################################################

# Run inference


tokens = tokenizer.tokenizer.encode(context)

insert_data(model, [], tokens, clear=True)

print("submitted", tokenizer.tokenizer.decode(tokens))

# bot.py


client = discord.Client(
    intents=discord.Intents.all())


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


@client.event
async def on_message(message):
    global tokens
    # print(
    #     f"message received({message.guild.name}:{message.channel.name}):", message.content)

    if message.author.bot:
        return

    if message.content == '+reset_rwkv':
        tokens = tokenizer.tokenizer.encode(context)
        insert_data(model, [], tokens, clear=True)
        await message.reply("RWKV has been reset")

    if message.content[:5] == '+rwkv':
        nnn = f"{endmessage}{name}{interface} {message.content[6:]}{endmessage}{bot}{interface}"
        newtokens = tokenizer.tokenizer.encode(nnn)
        print(f'add [{nnn}]')
        insert_data(model, tokens, newtokens)
        atokens = tokens + newtokens
        lenn = len(atokens)
        atokens = predict_next(atokens)
        while not (tokenizer.tokenizer.decode(atokens[-10:])[-len(endmessage):] == f"{endmessage}" or len(atokens) - lenn > 300):
            print(tokenizer.tokenizer.decode(atokens[-1]), end="")
            atokens = predict_next(atokens)
        send_msg = tokenizer.tokenizer.decode(
            atokens[lenn:])[:-len(endmessage)]
        print(f'send [{send_msg}]')
        await message.reply(send_msg)

        tokens = atokens
    if message.content[:7] == '+update':
        exit()
    if message.content[:8] == '+version':
        await message.channel.send("v0.3")
# get token from env
client.run(os.environ['TOKEN'])
