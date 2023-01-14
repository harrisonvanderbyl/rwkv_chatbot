import time
import torch

mymat = torch.randint(0, 255, (5000, 5000), dtype=torch.uint8)
mymat = mymat.cuda()

# warmup
for i in range(1000):
    mymat.to(dtype=torch.float32)


def myfunc(mat, dtype=torch.float32):
    return mat.to(dtype=dtype)


# time

timenow = time.time()
for i in range(1000):
    myfunc(mymat, dtype=torch.float32)

print(time.time() - timenow)

# time

timenow = time.time()
for i in range(1000):
    myfunc(mymat, dtype=torch.float16)

print(time.time() - timenow)

# time

timenow = time.time()
for i in range(1000):
    myfunc(mymat, dtype=torch.bfloat16)

print(time.time() - timenow)

# time

timenow = time.time()

for i in range(1000):
    myfunc(mymat, dtype=torch.float64)

print(time.time() - timenow)
