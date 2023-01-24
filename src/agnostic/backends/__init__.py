from src.agnostic.backends.torch import RWKVCudaOps, RWKVPTCompatOps, RWKVPTTSExportOps, RWKVSplitCudaOps, RWKVCudaQuantOps, RWKVStreamBigOps, RWKVCudaDeepspeedOps
from src.agnostic.backends.jax import RWKVJaxOps, RWKVNumpyOps
from src.agnostic.backends.tensorflow import RWKVTFExport, RWKVTFOps
from src.agnostic.backends.base import module
from typing import Dict
Backends: Dict[str, module] = {
    "tensorflow(cpu/gpu)": RWKVTFOps,
    "pytorch(cpu/gpu)": RWKVCudaOps,
    "numpy(cpu)": RWKVNumpyOps,
    "jax(cpu/gpu/tpu)": RWKVJaxOps,
    "pytorch-deepspeed(gpu)": RWKVCudaDeepspeedOps,
    "pytorch-quant(gpu-8bit)": RWKVCudaQuantOps,
    "pytorch-stream(gpu-config-vram)": RWKVStreamBigOps,
    "pytorch-split(2xgpu)": RWKVSplitCudaOps,
    "export-torchscript": RWKVPTTSExportOps,
    "export-tensorflow": RWKVTFExport,
    "pytorch-compatibility(cpu/debug)": RWKVPTCompatOps
}
