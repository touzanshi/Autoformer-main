import torch

a = torch.tensor([[1, 2, 3, 4]])
b = torch.tensor([[2, 3, 4, 5]])
c = [a, b]
print(torch.cat(c),'\n',torch.stack(c))
