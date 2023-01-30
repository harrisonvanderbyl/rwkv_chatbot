

import asyncio
from rwkvstic.rwkvMaster import RWKVMaster


async def runDiscordBot(model: RWKVMaster):
    import discord
    import os
    client = discord.Client(intents=discord.Intents.all())

    @client.event
    async def on_ready():
        print(f"Logged in as {client.user}")

    @client.event
    async def on_message(message: discord.Message):
        # check if author is bot
        if message.author.bot:
            return

        # check if message is a command
        if message.content.startswith("!rwkv "):
            mess = await message.channel.send("Loading...")
            model.resetState()
            model.loadContext(
                newctx=f"\n\nQuestion: {message.content[6:]}\n\nExpert Long Detailed Response: ")
            tex = ""
            for i in range(10):
                print(i)
                curr = model.forward(number=10)[
                    "output"]
                tex = tex + curr
                print(curr)

                if ("<|endoftext|>" in curr):
                    break
                mess = await mess.edit(content=tex)

            await asyncio.sleep(1)
            await mess.edit(content=tex)

        if message.content.startswith("!code"):
            text = message.content[6:]
            language = text.split(" ")[0]
            comment = text.split(" ")[1:]

            await message.channel.send("Loading...")

            model.resetState()
            model.loadContext(
                newctx=f"Q: Please write a {language} program that {comment}\n\nA:")
            model.loadContext(
                newctx=f"```Sure, here is a {language} program that {comment}:\n\n//{comment}\n")
            tex = ""
            for i in range(20):
                print(i)
                curr = model.forward(number=10, stopStrings=["</code>"])[
                    "output"]
                tex = tex + curr
                print(curr)
                if ("<|endoftext|>" in curr):
                    break
                mess = await mess.edit(content=tex)

            await asyncio.sleep(1)
            await mess.edit(content=tex)

    await client.start(os.environ.get("TOKEN", input("Discord Token:")))
