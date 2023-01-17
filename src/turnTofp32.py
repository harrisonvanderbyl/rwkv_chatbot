from sys import argv
import torch
from typing import Dict


Path = argv[0]

w: Dict[str, torch.Tensor] = torch.load(
    Path, map_location="cpu")

o = {}
keys = list(w.keys())
for x in keys:
    o[x] = w[x].float()
    del w[x]

torch.save(o, argv[1])
