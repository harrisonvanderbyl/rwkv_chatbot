import iree.compiler.tflite as iree_tflite_compile
import iree.runtime as iree_rt
import numpy
import os

import os
import torch
from src.utils import TOKENIZER
import inquirer
from torch.nn import functional as F
# context = 'A'
# context = "\nIn the"
# context = '\nSugar:'

files = os.listdir("tf")


questions = [
    inquirer.List('file',
                  message="What model do you want to use?",
                  choices=files,
                  ),
]
p = inquirer.prompt(questions)["file"]
loadFile = "tf/"+p


embed = int(loadFile.split("-")[2])
layers = int(loadFile.split("-")[1])
floatmode = (loadFile.split("-")[3])

if floatmode == "torch.float16":
    floatmode = torch.float16
elif floatmode == "torch.float32":
    floatmode = torch.float32
elif floatmode == "torch.bfloat16":
    floatmode = torch.bfloat16

emptyState = torch.load("onnx/"+p+"/emptyState.pt")

try:
    os.mkdir("iree")
except:
    pass

try:
    os.mkdir(
        f"iree/"+p)
except:
    pass


outpath = f"iree/"+p
d = os.listdir(outpath)
for f in d:
    os.remove(outpath+"/"+f)


questions = [
    inquirer.List('file',
                  message="What accelerator do you want to use?",
                  choices=["vmvx", "cuda", "vulkan-spirv", "llvm-cpu"],
                  ),
    inquirer.List('fp',
                  message="What fp doy ou want?",
                  choices=["16", "32"],
                  ),
]
q = inquirer.prompt(questions)

backends = [q["file"]]
config = "local-task"

iree_tflite_compile.compile_file(
    loadFile+f"/post/model_float{q['fp']}.tflite",
    input_type="TOSA",
    output_file=outpath+"/post.vmfb",
    save_temp_tfl_input=outpath+"/posttflite.mlir",
    save_temp_iree_input=outpath+"/posttosa.mlir",
    target_backends=backends,

    import_only=False)

iree_tflite_compile.compile_file(
    loadFile+f"/pre/model_float{q['fp']}.tflite",
    input_type="TOSA",
    output_file=outpath+"/pre.vmfb",
    save_temp_tfl_input=outpath+"/pretflite.mlir",
    save_temp_iree_input=outpath+"/pretosa.mlir",
    target_backends=backends,
    import_only=False)


def saveLayer(i, layer):
    print(f"Saving layer {i} {layer}")
    inpath = f"{loadFile}/layer-{str(i)}"
    iree_tflite_compile.compile_file(
        inpath+f"/model_float{q['fp']}.tflite",
        input_type="TOSA",
        output_file=outpath+f"/{str(i)}_{backends[0]}_.vmfb",
        save_temp_tfl_input=outpath+f"/{str(i)}tflite.mlir",
        save_temp_iree_input=outpath+f"/{str(i)}tosa.mlir",
        target_backends=backends,
        import_only=False)


layers = os.listdir(loadFile)
layers = filter(lambda x: "layer" in x, layers)
layers = list(layers)
layers.sort()
print(layers)


for i, layer in enumerate(layers):
    saveLayer(i, layer)
# config = iree_rt.Config("local-task")
# context = iree_rt.SystemContext(config=config)
# with open(outpath+"/post.vmfb", 'rb') as f:
#     vm_module = iree_rt.VmModule.from_flatbuffer(config.vm_instance, f.read())
#     context.add_vm_module(vm_module)

# invoke = context.modules.module["main"]
