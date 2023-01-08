from scipy.special import softmax
import numpy as np
import torch
import tqdm
from src.utils import TOKENIZER
TOKEN_MODE = "pile"
WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None
print(f'\nLoading tokenizer {WORD_NAME}...')
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)
if TOKEN_MODE == "pile":
    assert tokenizer.tokenizer.decode([187]) == '\n'


def loadContext(model, ctx: list[int], newctx: list[int], statex, progressCallBack=lambda x: x):

    with torch.jit.optimized_execution(True):
        for i in tqdm.tqdm(range(len(newctx))):

            x = ctx+newctx[:i+1]

            o = model.forward([x[-1]], statex)
            statex = o[1]
            progressCallBack(x)
    return ctx+newctx, o[1]


def sample(ozut, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
    try:
        ozut = ozut.numpy()
    except:
        ozut = np.array(ozut)
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # turn to float if is half and cpu
    probs = softmax(ozut, axis=-1)

    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p_usual)])
    probs[probs < cutoff] = 0
    if temp != 1.0:
        probs = pow(probs, 1.0 / temp)
    probs = probs / np.sum(probs, axis=0)
    mout = np.random.choice(a=len(probs), p=probs)
    return mout


class RWKVMaster():
    def __init__(self, model, emptyState, initTensor=lambda x: x):
        self.model = model
        self.tokenizer = tokenizer.tokenizer
        self.emptyState = emptyState
        self.myState = emptyState
        self.lastToken = 187
        self.initTensor = initTensor

    def forward(self, state=None, temp: float = 1.0, top_p_usual: float = 0.8):
        state = self.myState if state is None else state
        logits, state = self.model.forward([self.lastToken], state)
        self.myState = state
        sampled = sample(logits, temp, top_p_usual)
        self.lastToken = sampled
        sampled = self.tokenizer.decode([sampled])
        return {"logits": logits, "state": state, "output": sampled}

    def loadContext(self, ctx: str = "\n\n", newctx: str = "", statex=None, progressCallBack=lambda x: x):
        statex = self.myState if statex is None else statex
        ctx = self.tokenizer.encode(ctx)
        newctx = self.tokenizer.encode(newctx)
        ctx, state = loadContext(
            self.model, ctx, newctx, statex, progressCallBack)
        self.lastToken = ctx[-1]
        self.myState = state
        return ctx, state

    def sample(self, ozut, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
        return sample(ozut, temp, top_p_usual)

    def decode(self, x):
        return self.tokenizer.decode(x)

    def encode(self, x):
        return self.tokenizer.encode(x)

    def setState(self, state):
        self.myState = state

    def getState(self):
        return self.myState

    def resetState(self):
        self.myState = self.emptyState
