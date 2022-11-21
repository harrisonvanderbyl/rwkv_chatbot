import os
import torch
from src.utils import TOKENIZER
import inquirer
from torch.nn import functional as F
# context = 'A'
# context = "\nIn the"
# context = '\nSugar:'
import onnxruntime as ort
from onnx2tf import convert

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
files = os.listdir("onnx")


questions = [
    inquirer.List('file',
                  message="What model do you want to use?",
                  choices=files,
                  ),
]
loadFile = "onnx/"+inquirer.prompt(questions)["file"]


embed = int(loadFile.split("-")[2])
layers = int(loadFile.split("-")[1])
floatmode = (loadFile.split("-")[3])

if floatmode == "torch.float16":
    floatmode = torch.float16
elif floatmode == "torch.float32":
    floatmode = torch.float32
elif floatmode == "torch.bfloat16":
    floatmode = torch.bfloat16

emptyState = torch.load(loadFile+"/emptyState.pt")

try:
    os.mkdir("tf")
except:
    pass

try:
    os.mkdir(
        f"tf/rwkv-{int(emptyState.shape[0]/5)}-{emptyState.shape[1]}-{emptyState.dtype}")
except:
    pass

outpath = f"tf/rwkv-{int(emptyState.shape[0]/5)}-{emptyState.shape[1]}-{emptyState.dtype}"

# pre = onnx.load(f"{loadFile}/preprocess.onnx")  # load onnx model
# tf_pre = prepare(pre)  # prepare tf representation
# # export the model
# tf_pre.export_graph(
#     f"{outpath}/pre.pb")
convert(f"{loadFile}/preprocess.onnx",
        output_folder_path=f"{outpath}/pre", not_use_onnxsim=True)

# post = onnx.load(f"{loadFile}/postprocess.onnx")  # load onnx model
# tf_post = prepare(post)  # prepare tf representation
# tf_post.export_graph(f"{outpath}/pre.pb")  # export the model

convert(f"{loadFile}/postprocess.onnx",
        output_folder_path=f"{outpath}/post", not_use_onnxsim=True)

layers = os.listdir(loadFile)
layers = filter(lambda x: "layer" in x, layers)
layers = list(layers)
layers.sort()
print(layers)


def saveLayer(i, layer):
    print(f"Saving layer {i} {layer}")
    layer = convert(f"{loadFile}/{layer}",
                    output_folder_path=f"{outpath}/layer-{str(i)}", output_signaturedefs=True, not_use_onnxsim=True)
    # tf_layer = prepare(layer)  # prepare tf representation
    # tf_layer.export_graph(f"{outpath}/layer-{str(i)}.pb")  # export the model


for i, layer in enumerate(layers):
    saveLayer(i, layer)
