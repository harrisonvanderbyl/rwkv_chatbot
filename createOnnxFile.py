
import time
import gc
import torch
from src.utils import TOKENIZER
import os
from tqdm import tqdm

from torch.nn import functional as F
# context = 'A'
# context = "\nIn the"
# context = '\nSugar:'
import loadModelForOnnx
# context = "\n深圳是" # test Chinese
# context = "\n東京は" # test Japanese
pre, layers, post, emptyState = loadModelForOnnx.loadModel()
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


# tokens = loadContext(model, ctx=[], newctx=ctx1, statex=model.empty_state())


input_names = ["tokens", "state"]
output_names = ["probs", "outstate"]
try:
    os.mkdir("onnx")
except:
    pass

try:
    os.mkdir(
        f"onnx/rwkv-{int(emptyState.shape[0]/5)}-{emptyState.shape[1]}-{emptyState.dtype}")
except:
    pass
torch.save(
    emptyState, f"onnx/rwkv-{int(emptyState.shape[0]/5)}-{emptyState.shape[1]}-{emptyState.dtype}/emptyState.pt")

torch.onnx.export(pre, (torch.LongTensor([187])), f"onnx/rwkv-{int(emptyState.shape[0]/5)}-{emptyState.shape[1]}-{emptyState.dtype}/preprocess.onnx",
                  input_names=input_names[0:1], output_names=output_names[0:1], export_params=True, verbose=False, opset_version=int(os.environ.get("OPSET", "12")))

testOut = pre(torch.LongTensor([187]))

for m in range(len(layers)):
    torch.onnx.export(layers[m], (testOut, emptyState), f"onnx/rwkv-{int(emptyState.shape[0]/5)}-{emptyState.shape[1]}-{emptyState.dtype}/layer{m}.onnx",
                      input_names=output_names, output_names=output_names, export_params=True, verbose=False, opset_version=int(os.environ.get("OPSET", "12")))

torch.onnx.export(post, (testOut), f"onnx/rwkv-{int(emptyState.shape[0]/5)}-{emptyState.shape[1]}-{emptyState.dtype}/postprocess.onnx", input_names=output_names[:1],
                  output_names=output_names[:1], export_params=True, verbose=False, opset_version=int(os.environ.get("OPSET", "12")))