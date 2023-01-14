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
start = torch.Event(enable_timing=True)
end = torch.Event(enable_timing=True)
start.record()
for i in range(rounds):
    x = torch.mv(matbfloat, vecbfloat)

end.record()
torch.synchronize()
print(f"bfloat16 matmul: {start.elapsed_time(end)} ms")


# time the float matmul

start.record()
for i in range(rounds):
    x = torch.mv(mat, vec)

end.record()
torch.synchronize()
print(f"float matmul: {start.elapsed_time(end)} ms")

# time the double matmul

start.record()
for i in range(rounds):
    x = torch.mv(matdouble, vecdouble)

end.record()
torch.synchronize()
print(f"double matmul: {start.elapsed_time(end)} ms")
