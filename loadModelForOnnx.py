########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from genericpath import exists
from typing import List
from src.model_run_onnx import createRWKVModules, empty_state
import numpy as np
import math
import os
import sys
import types
import time
import gc
import torch
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
                                                            "fp32", "bf16", "fp16"],
                                                        )])["FLOAT_MODE"]

    args["CHUNK_SIZE"] = inquirer.text(
        message="What chunk size do you want to use?", default="128")

    args["CHUNK_SIZE"] = int(args["CHUNK_SIZE"])
    # print config
    print("RUN_DEVICE:", args["RUN_DEVICE"])
    print("FLOAT_MODE:", args["FLOAT_MODE"])
    print("")

    torch.set_num_threads(12)
    # opt
    # opt = inquirer.prompt([inquirer.List('opt',
    #                                      message="What jit mode do you want to use?",
    #                                      choices=[
    #                                          "script", "trace", "none"],
    #                                      )])["opt"]

    args["MODEL_NAME"] = file
    argsnums["ctx_len"] = 4068
    argsnums["vocab_size"] = vocab_size
    argsnums["head_qk"] = 0
    argsnums["pre_ffn"] = 0
    argsnums["grad_cp"] = 0
    argsnums["my_pos_emb"] = 0
    os.environ["RWKV_RUN_DEVICE"] = args["RUN_DEVICE"]

    if (torch.cuda.device_count() > 1):
        args["RUN_DEVICE"] = ["cuda:0"]*int(inquirer.text(
            message="Detected at least 2 Cuda Devices, how many chunks would you like to store on primary device, before offloading additional chunks?", default="20")) + ["cuda:1"]*20
    else:
        args["RUN_DEVICE"] = [args["RUN_DEVICE"]]*20

    ########################################################################################################
    # Step 2: set prompt & sampling stuffs
    ########################################################################################################
    pre, layers, post, n_layer = createRWKVModules(
        FloatMode=torch.float32 if args["FLOAT_MODE"] == "fp32" else torch.float16 if args["FLOAT_MODE"] == "fp16" else torch.bfloat16, Path=args["MODEL_NAME"], RunDevice=args["RUN_DEVICE"], chunkSize=args["CHUNK_SIZE"])

    emptyState = empty_state(pre.preProcess[1].shape[0], n_layer, torch.float32 if args["FLOAT_MODE"]
                             == "fp32" else torch.float16 if args["FLOAT_MODE"] == "fp16" else torch.bfloat16, args["RUN_DEVICE"])

    # if (opt == "script"):

    #     model = torch.jit.script(model)
    #     model = torch.jit.freeze(model)
    #     model = torch.jit.optimize_for_inference(model)
    # elif (opt == "trace"):
    #     model = torch.jit.trace(
    #         model, (torch.LongTensor([187]), model.empty_state()))
    #     model = torch.jit.freeze(model)
    #     model = torch.jit.optimize_for_inference(model)

    # layers = list(map(torch.jit.optimize_for_inference,
    #             map(lambda x: torch.jit.trace(x, (pre.forward([187]), emptyState)), layers)))
    # layers = list(map(torch.jit.optimize_for_inference, map(
    #     lambda x: torch.jit.script(x), layers)))

    return pre, layers, post, emptyState
