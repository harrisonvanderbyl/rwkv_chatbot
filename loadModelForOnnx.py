########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from genericpath import exists
from typing import List
from src.model_run_onnx import RWKV_RNN
import numpy as np
import math
import os
import sys
import types
import time
import gc
import torch
import torchdynamo
from src.utils import TOKENIZER
from tqdm import tqdm
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
import inquirer


def loadModel():
    files = os.listdir()
    # filter by ending in .pth
    files = [f for f in files if f.endswith(".pth")]

    questions = [
        inquirer.List('file',
                      message="What model do you want to use?",
                      choices=files,
                      ),
    ]
    file = inquirer.prompt(questions)["file"]

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    args = {}
    argsnums = {}

    ########################################################################################################
    # Step 1: set model & config
    # Do this first: pip install torchdynamo
    ########################################################################################################

    vocab_size = 50277

    # 'cpu' (already very fast) // 'cuda' // proc (faster then cpu, uses a fraction of the vram of cuda)
    args["RUN_DEVICE"] = inquirer.prompt([inquirer.List('RUN_DEVICE',
                                                        message="What device do you want to use?",
                                                        choices=[
                                                            "cpu", "cuda"],
                                                        )])["RUN_DEVICE"]

    # fp32 // bf16 (saves VRAM, slightly less accurate) // fp16 (saves VRAM, slightly less accurate, can only be used with cuda, sometimes faster)
    args["FLOAT_MODE"] = inquirer.prompt([inquirer.List('FLOAT_MODE',
                                                        message="What float mode do you want to use?",
                                                        choices=[
                                                            "fp32", "bf16", "fp16"] if args["RUN_DEVICE"] == "cuda" else ["fp32", "bf16"],
                                                        )])["FLOAT_MODE"]

    # print config
    print("RUN_DEVICE:", args["RUN_DEVICE"])
    print("FLOAT_MODE:", args["FLOAT_MODE"])
    print("")

    torch.set_num_threads(12)
    # opt
    opt = inquirer.prompt([inquirer.List('opt',
                                         message="What jit mode do you want to use?",
                                         choices=[
                                             "script", "trace", "none"],
                                         )])["opt"]

    args["MODEL_NAME"] = file
    argsnums["ctx_len"] = 4068
    argsnums["vocab_size"] = vocab_size
    argsnums["head_qk"] = 0
    argsnums["pre_ffn"] = 0
    argsnums["grad_cp"] = 0
    argsnums["my_pos_emb"] = 0
    os.environ["RWKV_RUN_DEVICE"] = args["RUN_DEVICE"]

    ########################################################################################################
    # Step 2: set prompt & sampling stuffs
    ########################################################################################################
    model = RWKV_RNN(args, argsnums)

    emptyState = model.empty_state()
    preprocess = model.preProcess[0]

    if (opt == "script"):

        model = torch.jit.script(model)
        model = torch.jit.freeze(model)
        model = torch.jit.optimize_for_inference(model)
    elif (opt == "trace"):
        model = torch.jit.trace(
            model, ((preprocess[187]), model.empty_state()))
        model = torch.jit.freeze(model)
        model = torch.jit.optimize_for_inference(model)

    return model, emptyState, preprocess
