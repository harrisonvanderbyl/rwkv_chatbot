import asyncio
import inquirer
from chat import runDiscordBot
from runWebsite import runWebsite
from src.rwkv import RWKV

services = inquirer.checkbox(
    message="What services do you want to use?",
    choices=[
        "Web Interface",
        "discord bot",
    ]
)
model = RWKV()

# run async simultaneously

funcs = []
if "Web Interface" in services:
    funcs.append(runWebsite(model))

if "discord bot" in services:
    funcs.append(runDiscordBot(model))


async def runAll():
    await asyncio.gather(*funcs)

asyncio.run(runAll())
