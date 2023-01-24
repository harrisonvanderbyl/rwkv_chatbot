from runWebsite import runWebsite
import asyncio
import inquirer
from chat import runDiscordBot
from rwkvstic.load import RWKV

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
