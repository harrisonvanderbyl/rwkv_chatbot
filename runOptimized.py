
import time
import gc
import torch
from rwkvstic.load import RWKV


# context = 'A'
# context = "\nIn the"
# context = '\nSugar:'
# context = "\n深圳是" # test Chinese
# context = "\n東京は" # test Japanese
# context = "\n深圳是" # test Chinese
# context = "\n東京は" # test Japanese


model = RWKV()

# Omodel = RWKV_RNN(q["file"])
# stat = None
# stat2 = emptyState
# print("testing diffs")
# for i in tqdm.tqdm(range(100)):
#     orig = Omodel.forward([145], stat)
#     orig = orig[0].cpu().numpy()

# for i in tqdm.tqdm(range(100)):
#     curr = model.forward([145], stat2)
#     stat2 = curr[1]
#     curr = curr[0].cpu().numpy()

# a = softmax(orig, axis=-1)
# b = softmax(curr, axis=-1)

# print("diffs", np.sum(np.abs(a-b)))


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

########################################################################################################


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


model.loadContext(ctx="\n\n", newctx=context)


for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
    print("--")
    time_ref = time.time_ns()

    if TRIAL == 0:

        gc.collect()
        torch.cuda.empty_cache()

    record_time('preprocess')
    text = context
    xout = 0.0
    with torch.no_grad():
        for i in range(100):

            char = model.forward()["output"]
            print(char, end='', flush=True)

            text += char

    record_time('total')

    # print(f'\n\n{time_slot}\n\n')
    print(
        f"\n\n--- preprocess {round(time_slot['preprocess'], 2)}s, generation {round(time_slot['total']-time_slot['preprocess'], 2)}s ", end=''
    )

print(("-" * 50) + '\n')
