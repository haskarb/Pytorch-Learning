import torch

x = torch.ones(4)
print(x)

#create a tensor in GPU
if torch.cuda.is_available():
    print("CUDA available")
    x = torch.ones(4, device="cuda")
    print(x)

