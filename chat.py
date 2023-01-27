

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

    await client.start(os.environ.get("TOKEN", input("Discord Token:")))
