import torch

mat = torch.tensor([[1.1, 2.2, 3.3], [3.3, 4.4, 1.4], [3.4, 4.3, 1.4]])
print(mat)
vec = torch.tensor([1., 2., 2.])
print(vec)
print(mat @ vec)
scale = torch.tensor([6.3, 1.2, 4.5])
print(scale)
print(mat * scale)
print(mat @ (vec * scale))
print((mat @ vec) * scale)
sc = mat@scale.diag()@mat.inverse()
print(sc)

print(sc@((mat) @ (vec)))
