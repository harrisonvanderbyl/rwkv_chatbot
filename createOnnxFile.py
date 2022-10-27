
import time
import gc
import torch
from src.utils import TOKENIZER
from tqdm import tqdm

from torch.nn import functional as F
# context = 'A'
# context = "\nIn the"
# context = '\nSugar:'
import loadModelForOnnx
# context = "\n深圳是" # test Chinese
# context = "\n東京は" # test Japanese
model = loadModelForOnnx.loadModel()
###### A good prompt for chatbot ######
context = '''
The '''
# context = "hello world! I am your supreme overlord!"
NUM_TRIALS = 999
LENGTH_PER_TRIAL = 200

TEMPERATURE = 1.0
top_p = 0.8
top_p_newline = 0.9  # only used in TOKEN_MODE = char

DEBUG_DEBUG = False  # True False --> show softmax output

########################################################################################################


print(model.n_layer)


print(f'\nOptimizing speed...')

gc.collect()
torch.cuda.empty_cache()

# input(0)

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

########################################################################################################


ctx1 = tokenizer.tokenizer.encode(context)
src_ctx1 = ctx1.copy()


print(
    "Note: currently the first run takes a while if your prompt is long, as we are using RNN to preprocess the prompt. Use GPT to build the hidden state for better speed.\n"
)

time_slot = {}
time_ref = time.time_ns()


def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt


init_out = []

out = []

print("torch.cuda.memory_allocated: %fGB" %
      (torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB" %
      (torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB" %
      (torch.cuda.max_memory_reserved(0)/1024/1024/1024))


def loadContext(self, ctx: list[int], statex: torch.Tensor, newctx: list[int]):
    for i in (range(len(newctx))):
        x = ctx+newctx[:i+1]
        o, statex = self.forward(
            torch.tensor(x), statex)
    return ctx+newctx, statex


tokens = loadContext(model, ctx=[], newctx=ctx1, statex=model.empty_state())


input_names = ["tokens", "state"]
output_names = ["output1"]

torch.onnx.export(model, (torch.tensor(tokens[0][-1:]), tokens[1]), f"rwkv-{model.n_layer}-{model.n_emb}-{model.FLOAT_MODE}.onnx", verbose=False,
                  input_names=input_names, output_names=output_names, export_params=True)
