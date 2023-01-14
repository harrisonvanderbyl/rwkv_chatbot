import time
import torch

mat = torch.randn(100, 100).float()
vec = torch.randn(100).float()
matbfloat = mat.bfloat16()
vecbfloat = vec.bfloat16()

matdouble = mat.double()
vecdouble = vec.double()

rounds = 100000

# warmup
for i in range(100):
    x = torch.mv(mat, vec)


# time the bfloat16 matmul


start = time.time()
for i in range(rounds):
    x = torch.mv(matbfloat, vecbfloat)

end = time.time()

print(f"bfloat16 matmul: {end-start} ms")


# time the float matmul

start = time.time()
for i in range(rounds):
    x = torch.mv(mat, vec)

end = time.time()

print(f"bfloat16 matmul: {end-start} ms")

# time the double matmul

start = time.time()
for i in range(rounds):
    x = torch.mv(matdouble, vecdouble)

end = time.time()

print(f"bfloat16 matmul: {end-start} ms")
