import torch
from typing import Dict
import inquirer
import os

Path = inquirer.prompt([inquirer.List('file',
                                      message="What model do you want to use?",
                                      choices=[
                                          f for f in os.listdir() if f.endswith(".pth")],
                                      )])["file"]

w: Dict[str, torch.Tensor] = torch.load(
    Path, map_location="cpu")

o = {}
keys = list(w.keys())
for x in keys:
    o[x] = w[x].float()
    del w[x]

torch.save(o, Path.replace(".pth", ".fp32.pth"))
