import requests
import discord
import os
import time
import gc
import torch

from src.rwkv import RWKV
from sty import Style, RgbFg, fg

fg.orange = Style(RgbFg(255, 150, 50))


model = RWKV()

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


print(f'\nOptimizing speed...')

gc.collect()
torch.cuda.empty_cache()


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


saveStates = {}
saveStates["empty"] = (model.emptyState.clone())

# Put the prompt into the init_state
init_state = model.loadContext("\n\n", context)
saveStates["questions"] = (init_state[1])


storys = []


@client.event
async def on_message(message):

    global model_tokens, currstate
    # print(
    #     f"message received({message.guild.name}:{message.channel.name}):", message.content)

    if message.author.bot:
        return

    msg = message.content.strip()

    temp = 1.0
    top_p = 0.8
    if ("-temp=" in msg):
        temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp="+str(temp), "")
        print(f"temp: {temp}")
    if ("-top_p=" in msg):
        top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p="+str(top_p), "")
        print(f"top_p: {top_p}")

    if msg == '+reset_drkv' or msg == '+drkv_reset':
        model.resetState()

        await message.reply(f"Chat reset. This is powered by RWKV-4 Language Model.")
        return

    if msg[:11] == '+drkv_save ':
        saveStates[msg[11:]] = model.getState()
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

    if msg.startswith('+drkv_load_website '):
        url = msg.split(' ')[1]
        print(f"Loading from {url}")
        r = requests.get(url)
        if r.status_code == 200:
            model.loadContext(
                "\n\n", "User: can you please read this document, and remember the imprtant bits?:\n\n\n"+r.text)
            await message.reply(f"Data loaded")
        else:
            await message.reply(f"Error loading from {url}")

    if msg[:6] == '+drkv ':

        real_msg = msg[6:].strip()
        new = f"\nUser: {real_msg}\n\nRWKV:"
        tknew = new
        print(f'### add ###\n[{new}]')

        currstate = model.loadContext("\n", tknew)
        begin = len(currstate[0])
        state = currstate

        with torch.no_grad():
            out = ""
            for i in range(100):
                o = model.forward()["output"]

                out += o

                if (o == "\n"):
                    break

        send_msg = out[len(tknew):]
        currstate = state
        if (len(send_msg) == 0):
            send_msg = "Error: No response generated."
        print(f'### send ###\n[{send_msg}]')
        await message.reply(send_msg)
    if msg[:10] == '+drkv_gen ':

        currstate = model.getState()

        real_msg = msg[10:].strip()
        new = f"{real_msg}".replace("\\n", "\n")
        tknew = new
        skiz = len(new)
        print(f'### add ###\n[{new}]')

        model.loadContext("\n", tknew, model.emptyState)

        with torch.no_grad():
            for i in range(100):
                o = model.forward()["output"]

                tknew += o

        send_msg = tknew[skiz:]
        state = model.getState()
        print(f'### send ###\n[{send_msg}]')
        await message.reply(send_msg+"\n continue with +drkv_cont "+str(len(storys)))
        storys.append(state)
        model.setState(currstate)

    if msg[:11] == '+drkv_cont ':
        real_msg = msg[11:].strip()
        oldstate = model.getState()
        state = storys[int(real_msg)]
        model.setState(state)

        with torch.no_grad():
            out = ""
            for i in range(100):
                o = model.forward()["output"]
                out += o

        send_msg = out
        print(f'### send ###\n[{send_msg}]')
        await message.reply(send_msg+"\n continue with +drkv_cont "+real_msg)
        storys[int(real_msg)] = model.getState()
        model.setState(oldstate)


client.run(os.environ["TOKEN"])
