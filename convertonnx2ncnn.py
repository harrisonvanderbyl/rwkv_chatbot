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
outpath = f"ncnn/rwkv-{int(layers)}-{embed}-{floatmode}"
try:
    os.mkdir("ncnn")
except:
    pass

try:
    try:
        os.rmdir(outpath)
    except:
        print("failed to delete path")
    os.mkdir(
        outpath)
except:
    pass


# pre = onnx.load(f"{loadFile}/preprocess.onnx")  # load onnx model
# ncnn_pre = prepare(pre)  # prepare tf representation
# # export the model
# tf_pre.export_graph(
#     f"{outpath}/pre.pb")
# os.system(
#     f"./onnx2ncnn {loadFile}/preprocess.onnx {outpath}/pre.param {outpath}/pre.bin")
# convert(f"{loadFile}/preprocess.onnx",
#         output_folder_path=f"{outpath}/pre", output_signaturedefs=True, not_use_onnxsim=True)

# post = onnx.load(f"{loadFile}/postprocess.onnx")  # load onnx model
# tf_post = prepare(post)  # prepare tf representation
# tf_post.export_graph(f"{outpath}/pre.pb")  # export the model

# convert(f"{loadFile}/postprocess.onnx",
#         output_folder_path=f"{outpath}/post", output_signaturedefs=True, not_use_onnxsim=True)

os.system(
    f"./onnx2ncnn {loadFile}/postprocess.onnx {outpath}/post.param {outpath}/post.bin")

layers = os.listdir(loadFile)
layers = filter(lambda x: "layer" in x, layers)
layers = list(layers)
layers.sort()
print(layers)


def saveLayer(i, layer):
    print(f"Saving layer {i} {layer}")
    os.system(
        f"./onnx2ncnn {loadFile}/{layer} {outpath}/layer-{str(i)}.param {outpath}/layer-{str(i)}.bin")
    # layer = convert(f"{loadFile}/{layer}",
    #                 output_folder_path=f"{outpath}/layer-{str(i)}", output_signaturedefs=True, not_use_onnxsim=True)
    # tf_layer = prepare(layer)  # prepare tf representation
    # tf_layer.export_graph(f"{outpath}/layer-{str(i)}.pb")  # export the model


for i, layer in enumerate(layers):
    saveLayer(i, layer)
