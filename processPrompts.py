import tqdm
from sys import argv
from src.rwkv import RWKV
import json
import os
model = RWKV()
dataset = argv[-1]

# open jsonl
jsonl = open(dataset).readlines()
jsonl = [json.loads(line)["question"] for line in jsonl]
# check file exists
os.system("ls output.jsonl")


otputs = []
with open("output.jsonl", "w") as f:
    for x in tqdm.tqdm(jsonl):
        model.resetState()
        o, start = model.loadContext("\n",
                                     f"Prompt: {x}?\n\nLong Detailed Expert Response: ")

        output = model.forward(number=100)["output"]
        f.write(json.dumps({"question": x, "LDER": output})+"\n")
