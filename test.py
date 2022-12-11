import torch
from torch.profiler import profile, record_function, ProfilerActivity
mat = torch.rand(3, 3)
# print(mat)
vec = torch.rand(3)
dvec = torch.rand(3)
# print(vec)

testt = mat@mat
testa = torch.einsum('ij,jk->ik', [mat, mat])
testt = mat@vec
testa = torch.einsum('ij,j->i', [mat, vec])
testb = torch.sum(mat*vec, 1)
# testb = torch.sum(mat*mat, 1)
# print(testt)
# print(testa)
# print(testt)
# print(testa)
# print(testb)

# avv = torch.exp(mat@(vec+dvec))
# print(avv)
# avr = torch.exp(mat@(vec) + mat@(dvec))
# print(avr)
# avx = torch.exp(mat@vec) * torch.exp(mat@dvec)
# print(avx)
# avy = torch.exp(torch.sum(mat*vec, 1)) * torch.exp(mat@dvec)
# print(avy)
# avz = torch.exp(torch.sum(mat*vec, 1)) * torch.exp(torch.sum(mat*dvec, 1))
# print(avz)
# optimise this
# avw = exp(a+b) = exp(a)*exp(b)
# log(exp(a)*exp(b)) = log(exp(a)) + log(exp(b))
# print(avw)
# testb = torch.sum(mat*mat, 1)
# rounds = 1000

# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     with record_function("mat@vec"):
#         for x in range(rounds):
#             testt = mat@vec
#     with record_function("einsum(ij,j->i)"):
#         for x in range(rounds):
#             testa = torch.einsum('ij,j->i', [mat, vec])
#     with record_function("torch.sum(mat*vec,1)"):
#         for x in range(rounds):
#             testa = torch.sum(mat*vec, 1)
#     with record_function("torch.matmul(mat,vec)"):
#         for x in range(rounds):
#             testt = torch.matmul(mat, vec)
#     with record_function("torch.mv(mat,vec)"):
#         for x in range(rounds):
#             testt = torch.mv(mat, vec)
# print(prof.key_averages().table(sort_by="cpu_time_total",
#       top_level_events_only=True, row_limit=10))

rounds = 1000

matb = mat.to(torch.bfloat16)
vecb = vec.to(torch.bfloat16)
matp = mat.to(torch.float64)
vecp = vec.to(torch.float64)
matu = mat.to(torch.uint8)
vecu = vec.to(torch.uint8)
math = mat.to(torch.half)
vech = vec.to(torch.half)

# with cuda
matbc = matb.cuda()
vecbc = vecb.cuda()
matpc = matp.cuda()
vecpc = vecp.cuda()
matuc = matu.cuda()
vecuc = vecu.cuda()
matc = mat.cuda()
vecc = vec.cuda()
mathc = math.cuda()
vechc = vech.cuda()


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("fp32"):
        for x in range(rounds):
            testt = torch.mv(mat, vec)
    with record_function("bf16"):
        for x in range(rounds):
            testt = torch.mv(matb, vecb)
    with record_function("fp64"):
        for x in range(rounds):
            testt = torch.mv(matp, vecp)
    # with record_function("uint8"):
    #     for x in range(rounds):
    #         testt = torch.mv(matu, vecu)
    with record_function("fp32 cuda"):
        for x in range(rounds):
            testt = torch.mv(matc, vecc)
    with record_function("bf16 cuda"):
        for x in range(rounds):
            testt = torch.mv(matbc, vecbc)
    with record_function("fp64 cuda"):
        for x in range(rounds):
            testt = torch.mv(matpc, vecpc)
    with record_function("half cuda"):
        for x in range(rounds):
            testt = torch.mv(mathc, vechc)

print(prof.key_averages().table(sort_by="cuda_time_total",
      top_level_events_only=True, row_limit=10))
