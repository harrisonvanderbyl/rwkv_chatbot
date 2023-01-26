

from rwkvstic.rwkvMaster import RWKVMaster


async def runDiscordBot(model: RWKVMaster):
    import discord
    import os
    client = discord.Client(intents=discord.Intents.all())

    @client.event
    async def on_ready():
        print(f"Logged in as {client.user}")

    @client.event
    async def on_message(message):
        # check if author is bot
        if message.author.bot:
            return

        # check if message is a command
        if message.content.startswith("!rwkv "):
            mess = await message.reply("Generating...")
            model.resetState()
            model.loadContext(
                newctx=f"\n\nQuestion: {message.content[6:]}\n\nExpert Long Detailed Response: \n\n")
            tex = ""
            for i in range(100):
                tex = tex + model.forward()[
                    "output"]
                if tex.endswith(("\n\n", "<|endoftext|>")):
                    break
                mess.edit(content=tex)

    await client.start(os.environ.get("TOKEN", input("Discord Token:")))
