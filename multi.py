import inquirer
from runWebsite import runWebsite
from src.rwkv import RWKV

services = inquirer.prompt([inquirer.checkbox('service',
                                              message="What services do you want to use?",
                                              choices=[
                                                  "Web Interface",
                                                  "discord bot",
                                              ]
                                              )])["service"]
model = RWKV()
if "Web Interface" in services:
    runWebsite(model)

if "discord bot" in services:
    runDiscordBot()
