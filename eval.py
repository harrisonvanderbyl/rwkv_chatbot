# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json
import inquirer
import torch.nn.functional as F
from tqdm import tqdm
from lm_eval.base import CacheHook
from lm_eval.models.gpt2 import GPT2LM
from lm_eval import tasks, evaluator, utils

import numpy as np
import math
import os
import datetime
# os.environ["CUDA_VISIBLE_DEVICES"] = '7' # CHANGE ME!
import torch
from torch.nn import functional as F

from src.rwkvops import RwkvOpList
from src.utils import TOKENIZER
from sty import Style, RgbFg, fg

fg.orange = Style(RgbFg(255, 150, 50))

RWKV_PAD = [0]  # <|endoftext|>
# RWKV_PAD = [187] # \n
# RWKV_PAD = [187, 187] # \n\n

RUN_TABLE = [1652]  # part of model file name
RUN_MODEL_NAME = '/mnt/ssd-1/BlinkDL_dont_delete/B/TRAIN_100M/out/all-'

# eval_tasks = ['lambda']
# eval_tasks = ['hellaswag']
eval_tasks = ['piqa']

TEST_MODEL = 'rwkv'  # 'rwkv' 'neo'
USE_CUDA = True  # True False
RUN_DEVICE = 'cuda' if USE_CUDA else 'cpu'  # cpu cuda
# Set RUN_DEVICE in src/model.py too !!!

RWKV_SLOW_MODE = True  # True False


class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = 0

    def encode(self, string: str, add_special_tokens=False):
        return self.tokenizer.encode(string)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


class EvalHarnessAdapter(GPT2LM):
    def __init__(self):
        TOKEN_MODE = "pile"
        WORD_NAME = [
            "20B_tokenizer.json",
            "20B_tokenizer.json",
        ]  # [vocab, vocab] for Pile model
        UNKNOWN_CHAR = None
        print(f'\nLoading tokenizer {WORD_NAME}...')
        tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)
        assert tokenizer.tokenizer.decode([187]) == '\n'

        self.tokenizer = TokenizerWrapper(
            tokenizer=tokenizer.tokenizer)

    def greedy_until(self, requests):
        raise NotImplementedError()

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []
        sum_logit = 0
        nCorrect = 0

        for COUNTER in (range(len(requests))):
            n = COUNTER

            raw_src = requests[n][0][0] + requests[n][0][1]

            src = requests[n][1] + requests[n][2]
            if TEST_MODEL == 'rwkv':
                raw_src = '\n' + raw_src
                src = RWKV_PAD + src

            sss = str(src)
            correct = True
            if sss in logitBuf:
                logit = logitBuf[sss]
                correct = correctBuf[sss]
            else:
                q_len = len(requests[n][1])
                if TEST_MODEL == 'rwkv':
                    q_len += len(RWKV_PAD)
                logit = 0

                with torch.no_grad():
                    if RWKV_SLOW_MODE:
                        state = None
                        fg.orange = Style(RgbFg(255, 255, 255))
                        print(fg.orange+"\n", end='')

                        for i in (range(1, len(src))):
                            x = src[:i]

                            print(self.tokenizer.decode([x[-1]]), end='')
                            fg.orange = Style(RgbFg(255, 255, 255))
                            out, state = rwkv_rnn.forward(x, state)
                            if i >= q_len:
                                oo = torch.tensor(out)
                                sorted_probs, s_index = torch.sort(
                                    oo, descending=True)
                                pred = s_index[0].item()
                                if pred != src[i]:
                                    correct = False
                                    fg.orange = Style(RgbFg(255, 0, 0))
                                else:
                                    fg.orange = Style(RgbFg(0, 255, 0))
                                print(fg.orange, end="")
                                # print(x, '=>', src[i], 'pred', pred)
                                logit += math.log(F.softmax(oo,
                                                  dim=-1)[src[i]])

                logitBuf[sss] = logit
                correctBuf[sss] = correct

            if correct:
                nCorrect += 1
            res += [(logit, correct)]
            sum_logit += logit
            mean = sum_logit / (COUNTER+1)
            acc = nCorrect / (COUNTER+1) * 100

            if n % 100 == 0:
                print(f'{n//100}/{len(requests)//100}', end=' ', flush=True)
        return res

    @torch.no_grad()
    def run_eval(self, eval_tasks=None, num_fewshot=0, bootstrap_iters=2):
        results = evaluator.evaluate(
            lm=self,
            task_dict=tasks.get_task_dict(eval_tasks),
            provide_description=False,
            num_fewshot=num_fewshot,
            limit=None,
            bootstrap_iters=bootstrap_iters,
        )
        return results


RWKV_ID = ''
for RUN_NUM in RUN_TABLE:
    RWKV_ID = RUN_NUM
    logitBuf = {}
    correctBuf = {}
    files = os.listdir()
    # filter by ending in .pth
    files = [f for f in files if f.endswith(".pth")]

    questions = [
        inquirer.List('file',
                      message="What model do you want to use?",
                      choices=files,
                      ),
        inquirer.List(
            'method',
            message="What inference method?",
            choices=RwkvOpList.keys()
        )

    ]
    q = inquirer.prompt(questions)
    RWKV_FILENAME = q["file"]

    if RWKV_SLOW_MODE:
        # from src.model_run import RWKV_RNN
        # rwkv_rnn = RWKV_RNN(RWKV_FILENAME)
        from src.rwkv import RWKV
        rwkv_rnn, emptystate = RWKV(RWKV_FILENAME, q["method"])

    import tokenizers

    print("Running evaluation harness...")
    adapter = EvalHarnessAdapter()
    results = adapter.run_eval(
        eval_tasks=eval_tasks,
        bootstrap_iters=10000,
    )
    print(results)
    # output results to file
    with open(f"res.json", "w") as f:
        json.dump(results, f)
