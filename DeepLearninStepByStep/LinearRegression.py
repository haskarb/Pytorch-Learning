# Linear Regression with PyTorch

# # imports
# import torch
# import torch.nn as nn
# import numpy as np
# import torch.optim as optim

# from torchviz import make_dot


# -*- coding: utf-8 -*-

import torch

if __name__ == '__main__':
    print(torch.backends.cuda.is_built())
    print(torch.backends.cudnn.is_available())
    print(torch.backends.cudnn.version())
    print(torch.backends.cudnn.enabled)
    print(torch.backends.cudnn.allow_tf32)
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())
    print(torch.backends.openmp.is_available())
    print(torch.backends.mkldnn.is_available())