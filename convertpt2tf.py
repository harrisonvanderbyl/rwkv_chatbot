from tqdm import tqdm
from src.utils import TOKENIZER
from scipy.special import softmax
import torch
import numpy as np
import src.model_run_onnx as mro
import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')


tfl, emptyState = mro.createRWKVTensorflowModel(
    "./RWKV-3-Pile-20220720-10704.pth")

tf.saved_model.save(tfl, "./tensorflow-models/RWKV-mini",
                    signatures={'my_signature': tfl.forward.get_concrete_function()})

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [tfl.forward.get_concrete_function()])
tflite_model = converter.convert()
with open("./tensorflow-models/RWKV-mini/lite32.tflite", 'wb') as f:
    f.write(tflite_model)

# # create bf16
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open("./tensorflow-models/RWKV-mini/lite16.tflite", 'wb') as f:
    f.write(tflite_model)
