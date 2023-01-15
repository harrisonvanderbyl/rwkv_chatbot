import time
import torch

mymat = torch.rand(0, 2, 1000, dtype=torch.float16)
mymatb = mymat.bfloat16()

# warmup
for i in range(10000):
    mymat.to(dtype=torch.float32)


def myfunc(mat, dtype=torch.float32):
    return mat.to(dtype=dtype)


# time

timenow = time.time()
for i in range(100000):
    myfunc(mymat, dtype=torch.float32)

print(time.time() - timenow)

# time

timenow = time.time()
for i in range(100000):
    myfunc(mymatb, dtype=torch.float32)

print(time.time() - timenow)
