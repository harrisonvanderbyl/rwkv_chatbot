import torch

mat = torch.randn(1000, 1000).cuda()/10
vec = torch.randn(1000).cuda()/10

# warmup
for i in range(100):
    mat @ vec

# baseline
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for i in range(1000):
    mat.mv(vec).exp()

end.record()
torch.cuda.synchronize()

print(mat.mv(vec).exp()[0])
print(f"baseline: {start.elapsed_time(end)} ms")

# experimental
start.record()
for i in range(1000):
    (mat*vec).exp().prod(1)

end.record()
torch.cuda.synchronize()
print(f"experimental: {start.elapsed_time(end)} ms")
# print((mat*vec).exp().prod(1)[0])

# experimental
axpmat = mat.exp()
vecexp = vec.exp()

start.record()
for i in range(1000):
    vecexp.pow(mat).prod(1)

end.record()
torch.cuda.synchronize()
print(f"experimental: {start.elapsed_time(end)} ms")
# print(vecexp.pow(mat).prod(1)[0])

# experimental
start.record()
for i in range(1000):
    axpmat.pow(vec).prod(1)

end.record()
torch.cuda.synchronize()
print(f"experimental: {start.elapsed_time(end)} ms")

# experimental
start.record()
for i in range(1000):
    axpmat.prod(1)
