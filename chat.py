

from rwkvstic.rwkvMaster import RWKVMaster

states = {}


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
            model.loadContext(newctx=f"Q: {message.content[6:]}\n\nA: ")
            text = model.forward(stopStrings=["<|endoftext|>"], number=100)[
                "output"]
            states[message.id] = model.getState()
            await message.reply(text)

        # check if message is a reply
        if message.reference is not None:
            if message.reference.message_id in states.keys():
                model.setState(states[message.reference.message_id])
                model.loadContext(newctx=f"\n\nQ: {message.content}\n\nA: ")
                text = model.forward(stopStrings=["<|endoftext|>"], number=100)[
                    "output"]
                states[message.id] = model.getState()
                await message.reply(text)
    await client.start(os.environ.get("TOKEN", input("Discord Token:")))
