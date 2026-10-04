import os
import torch
torch.cuda.init()
print("CUDA init OK")
x = torch.randn(3, 3).cuda()
print("GPU tensor OK:", x.device)
print("If you see this, CUDA works from this file")
