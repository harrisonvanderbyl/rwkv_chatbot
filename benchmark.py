########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from tqdm import tqdm
from rwkvstic.load import RWKV
from torch.nn import functional as F
import torch
import os
import sys
import types
import json
import math
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
with open(f"misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " +
             doc['text'].rsplit(' ', 1)[1]] for doc in todo]
args = types.SimpleNamespace()

########################################################################################################


PAD_SEQ = "\n"

########################################################################################################

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


model = RWKV()

print('Running...')
xsum = 0
xcnt = 0
xacc = 0
for d in tqdm(todo):
    model.resetState()
    src = PAD_SEQ + d[0]
    dst = model.tokenizer.encode(d[1])

    logits = 0
    correct = True
    for i in range(len(dst)):
        if i == 0:
            model.loadContext(newctx=src)
        else:
            model.lastToken = dst[i-1]
        out = model.forward()["logits"]
        probs = F.softmax(out.float(), dim=-1)
        logits += math.log(probs[dst[i]])
        _, s_index = torch.sort(probs, descending=True)
        pred = s_index[0].item()
        if pred != dst[i]:
            correct = False

    xcnt += 1
    xsum += logits
    xacc += 1 if correct else 0
    if xcnt % 100 == 0 or xcnt == len(todo):
        print("\n\n")
        print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc',
              round(xacc/xcnt*100, 2))  # , 'pred', pred, 'dst', dst)
