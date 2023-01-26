

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
            model.resetState()
            model.loadContext(
                newctx=f"\n\nQuestion: {message.content[6:]}\n\nExpert Long Detailed Response: ")
            text = model.forward(stopStrings=["<|endoftext|>"], number=100)[
                "output"]
            await message.reply(text)

    await client.start(os.environ.get("TOKEN", input("Discord Token:")))
