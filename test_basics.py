import torch
x = torch.randn(20)
print(x)
# Get the indices that would sort x (in terms of abs value)
indices = torch.argsort(x.abs(),descending=True)
print(indices)
# choose the first 10 
indices_smallest = indices[5:]

# Zero those positions out
x[indices_smallest] = 0.0
print(x)