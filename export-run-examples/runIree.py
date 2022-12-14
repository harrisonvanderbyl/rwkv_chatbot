import numpy
import iree.runtime as iree_rt
from genericpath import exists
from typing import List
import numpy as np
import os
import time
import gc
import torch
from src.utils import TOKENIZER
import inquirer
from torch.nn import functional as F
# context = 'A'
# context = "\nIn the"
# context = '\nSugar:'
from typing import List as list
import tqdm
# context = "\n深圳是" # test Chinese
# context = "\n東京は" # test Japanese


files = os.listdir("iree")


questions = [
    inquirer.List('file',
                  message="What model do you want to use?",
                  choices=files,
                  ),
]
loadFile = "iree/"+inquirer.prompt(questions)["file"]


embed = int(loadFile.split("-")[2])
layernum = int(loadFile.split("-")[1])
floatmode = (loadFile.split("-")[3])

if floatmode == "torch.float16":
    floatmode = torch.float16
elif floatmode == "torch.float32":
    floatmode = torch.float32
elif floatmode == "torch.bfloat16":
    floatmode = torch.bfloat16

# # emptyState = torch.load(loadFile+"/emptyState.pt")
# print(emptyState.shape)


layers = os.listdir(loadFile)
layers = filter(
    lambda x: not "pre" in x and not 'post' in x and not "mlir" in x, layers)
layers = list(layers)
layers.sort()
print(layers)
dname = layers[0].split("_")[1]
if (dname == "llvm-cpu"):
    dname = "local-task"
if (dname == "vulkan-spirv"):
    dname = "vulkan"


def loadModel(x):

    config = iree_rt.Config(driver_name=dname)
    context = iree_rt.SystemContext(config=config)
    with open(loadFile+"/"+x+".vmfb", 'rb') as f:
        vm_module = iree_rt.VmModule.from_flatbuffer(
            config.vm_instance, f.read())
        context.add_vm_module(vm_module)
    return context.modules.module["main"]


pre = loadModel("pre")
post = loadModel("post")

layers = list(map(lambda x: loadModel(x[1][:-5]), enumerate(layers)))
###### A good prompt for chatbot ######
context = '''
The following is a conversation between a highly knowledgeable and intelligent AI assistant, called RWKV, and a human user, called User. In the following interactions, User and RWKV will converse in natural language, and RWKV will do its best to answer User’s questions. RWKV was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. The conversation begins.

User: OK RWKV, I’m going to start by quizzing you with a few warm-up questions. Who is currently the president of the USA?

RWKV: It’s Joe Biden; he was sworn in earlier this year.

User: What year was the French Revolution?

RWKV: It started in 1789, but it lasted 10 years until 1799.

User: Can you guess who I might want to marry?

RWKV: Only if you tell me more about yourself - what are your interests?

User: Aha, I’m going to refrain from that for now. Now for a science question. What can you tell me about the Large Hadron Collider (LHC)?

RWKV: It’s a large and very expensive piece of science equipment. If I understand correctly, it’s a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.
'''
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


def loadContext(ctx: list[int], state, newctx: list[int]):
    state = numpy.array(state, dtype=numpy.float32)
    for i in tqdm.tqdm(range(len(newctx))):
        x = ctx+newctx[:i+1]
        o = pre(numpy.array([x[-1]], dtype=numpy.int32))

        x = o.to_host()

        for l in layers:

            state, x = l(x, state)

    return ctx+newctx, state.to_host()


tokens = loadContext(ctx=[], newctx=ctx1, state=((layernum*5)*[embed*[0]]))


def sample_logits(ozut: torch.Tensor, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # turn to float if is half and cpu
    probs = F.softmax(ozut.float(), dim=-1)

    sorted_probs = torch.sort(probs, descending=True)[0]
    cumulative_probs = torch.cumsum(
        sorted_probs.float(), dim=-1).cpu().numpy()
    cutoff = float(sorted_probs[np.argmax(
        cumulative_probs > top_p_usual)])
    probs[probs < cutoff] = 0
    if temp != 1.0:
        probs = probs.pow(1.0 / temp)

    out: int = torch.multinomial(probs.float(), 1, True)[0]
    return out


for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
    print("--")
    time_ref = time.time_ns()

    if TRIAL == 0:

        gc.collect()
        torch.cuda.empty_cache()

    record_time('preprocess')
    with torch.no_grad():
        for i in range(100):
            chars: List[int] = tokens[0]

            statex = tokens[1]
            x = pre(numpy.array([chars[-1]], dtype=numpy.int32)).to_host()

            for l in layers:
                statex, x = l(x, statex)

            myout = (post(x.to_host()).to_host(), statex)

            chars += [sample_logits(
                torch.Tensor(myout[0]), temp=TEMPERATURE, top_p_usual=top_p)]
            char = tokenizer.tokenizer.decode(chars[-1])

            tokens = (chars, myout[1])

            if '\ufffd' not in char:
                print(char, end="", flush=True)

    record_time('total')
    # print(f'\n\n{time_slot}\n\n')
    print(
        f"\n\n--- preprocess {round(time_slot['preprocess'], 2)}s, generation {round(time_slot['total']-time_slot['preprocess'], 2)}s ", end=''
    )

print(("-" * 50) + '\n')
