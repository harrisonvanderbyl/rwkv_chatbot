import torch
from torch.profiler import profile, record_function, ProfilerActivity
mat = torch.rand(100, 100)
print(mat)
vec = torch.rand(100)
print(vec)

testt = mat@mat
testa = torch.einsum('ij,jk->ik', [mat, mat])
testt = mat@vec
testa = torch.einsum('ij,j->i', [mat, vec])
testb = torch.sum(mat*vec, 1)
# testb = torch.sum(mat*mat, 1)
print(testt)
print(testa)
print(testt)
print(testa)
print(testb)
# testb = torch.sum(mat*mat, 1)
rounds = 1000

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("mat@vec"):
        for x in range(rounds):
            testt = mat@vec
    with record_function("einsum(ij,j->i)"):
        for x in range(rounds):
            testa = torch.einsum('ij,j->i', [mat, vec])
    with record_function("torch.sum(mat*vec,1)"):
        for x in range(rounds):
            testa = torch.sum(mat*vec, 1)
    with record_function("torch.matmul(mat,vec)"):
        for x in range(rounds):
            testt = torch.matmul(mat, vec)
    with record_function("torch.mv(mat,vec)"):
        for x in range(rounds):
            testt = torch.mv(mat, vec)
print(prof.key_averages().table(sort_by="cpu_time_total",
      top_level_events_only=True, row_limit=10))
