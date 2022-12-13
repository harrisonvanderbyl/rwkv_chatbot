########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from functools import reduce
from genericpath import exists
from typing import List
from src.model_run import isIn, sample
from src.model_run_onnx import createRWKVModules, empty_state
import numpy as np
import math
from torch.nn import functional as F
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


def loadModel(trace=False, compat=False):
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
    args["FLOAT_MODE"] = inquirer.prompt([inquirer.List('RUN_DEVICE',
                                                        message="What device do you want to use?",
                                                        choices=[
                                                            "fp16", "bf16", "fp32", "fp64"],
                                                        )])["RUN_DEVICE"]

    args["CHUNK_SIZE"] = inquirer.text(
        message="What chunk size do you want to use?", default="4")

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

    if (torch.cuda.device_count() > 1 and args["RUN_DEVICE"] == "cuda"):
        args["RUN_DEVICE"] = ["cuda:0"]*int(inquirer.text(
            message="Detected at least 2 Cuda Devices, how many chunks would you like to store on primary device, before offloading additional chunks?", default="20")) + ["cuda:1"]*50
    else:
        if args["RUN_DEVICE"] == "cuda":
            args["RUN_DEVICE"] = ["cuda"]*int(inquirer.text(
                message="how many chunks would you like to store on primary device, before streaming additional chunks?", default="20")) + ["stream"]*50
        else:
            args["RUN_DEVICE"] = ["cpu"]*50
    ########################################################################################################
    # Step 2: set prompt & sampling stuffs
    ########################################################################################################

    intmode = inquirer.prompt([inquirer.List('INTMODE',
                                             message="What int mode do you want to use?",
                                             choices=[
                                                 "int32", "int64"],
                                             )])["INTMODE"]
    if (intmode == "int32"):
        intmode = torch.int32
    else:
        intmode = torch.int64

    pre, layers, post, n_layer = createRWKVModules(
        FloatMode=torch.float64 if args["FLOAT_MODE"]
        == "fp64" else torch.float32 if args["FLOAT_MODE"] == "fp32" else torch.float16 if args["FLOAT_MODE"] == "fp16" else torch.bfloat16, Path=args["MODEL_NAME"], RunDevice=args["RUN_DEVICE"], chunkSize=args["CHUNK_SIZE"], inttype=intmode, compat=compat)

    emptyState = empty_state(pre.preProcess[1].shape[0], n_layer,  torch.float64 if args["FLOAT_MODE"]
                             == "fp64" else torch.float32 if args["FLOAT_MODE"]
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

    # layers = list(map(torch.jit.optimize_for_inference, map(
    #     lambda x: torch.jit.script(x), layers)))
    if (trace and args["RUN_DEVICE"][0] != "stream"):
        pret: torch.ScriptModule = torch.jit.trace(pre, example_inputs=(
            torch.Tensor([187]).to(dtype=torch.int32, device=args["RUN_DEVICE"][0]), emptyState), check_trace=False)

        layerst: list[torch.ScriptModule] = list(map(lambda x: torch.jit.trace(
            x, example_inputs=pre.forward(torch.LongTensor([187]), emptyState), check_trace=False), layers))

        postt: torch.ScriptModule = torch.jit.trace(
            post, example_inputs=pre.forward(torch.LongTensor([187]), emptyState), check_trace=False)

        return pret, layerst, postt, emptyState
    else:
        # layers = list(map(lambda x: torch.compile(x), layers))
        # pre = torch.compile(pre)
        # post = torch.compile(post)
        return pre, layers, post, emptyState


class Compat():
    def __init__(self, pre, layers, post, emptyState):
        self.pre = pre
        self.layers = layers
        self.post = post
        self.emptyState = emptyState

    def loadContext(self, ctx: list[int] = [], newctx: list[int] = [], statex=None, silent=False):
        if (statex == None):
            statex = self.emptyState
        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        #     with record_function("model_inference"):
        with torch.jit.optimized_execution(True):
            for i in tqdm(range(len(newctx))) if not silent else range(len(newctx)):

                x = ctx+newctx[:i+1]
                o = self.pre.forward(torch.LongTensor([x[-1]]), statex)

                for s in self.layers:
                    o = s.forward(*o)
                statex = o[1]
                # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        #     with record_function("model_inference"):
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        return ctx+newctx, o[1]

    def sample_logits(self, ozut: torch.Tensor, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
        out = ozut
        if out.dtype == torch.half and out.device == torch.device('cpu'):
            out = out.float()
        probs = F.softmax(out, dim=-1)

        return sample(probs, temperature=temp, top_p_usual=top_p_usual)

    def run(self, currstate: list({"score": float, "ctx": list, "state": torch.Tensor}), temp: float = 1.5, top_p: float = 0.9, nla: float = 0, endChars=[[187, 187], [535]]):
        options = []

        chars = currstate[0]["ctx"]
        # if any(list(map(lambda x: x == ctx[-len(x):], endChars))):
        #     return options

        state = currstate[0]["state"]

        x = self.pre.forward(torch.LongTensor([chars[-1]]), state)

        for l in self.layers:
            x = l.forward(*x)

        xout = self.post.forward(*x)
        chars += [self.sample_logits(
            xout[0], temp=0.9, top_p_usual=0.8)]

        tokens = [{"ctx": chars, "state": xout[1], "score": 1.0}]

        return tokens

    def empty_state(self):
        return self.emptyState


def loadModelCompat():
    pre, layers, post, emptyState = loadModel()
    return Compat(pre, layers, post, emptyState)
