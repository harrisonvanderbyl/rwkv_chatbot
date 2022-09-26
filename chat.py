########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import os
import discord
import subprocess
print("start")


client = discord.Client(
    intents=discord.Intents.all())


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


@client.event
async def on_message(message):

    print(
        f"message received({message.guild.name}:{message.channel.name}):", message.content)
    if message.author.bot:
        return
    if message.content[:7] == '+update':
        exit()
    if message.content[:8] == '+version':
        await message.channel.send("v0.1")
# get token from env
client.run(os.environ['TOKEN'])
