import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets,transforms
import random
from torch import optim,ceil
import torchvision.models as models
import partition
import torch.nn.functional as F
from torch import nn
def Model():
    model=models.mobilenet_v2(pretrained=True)
    return model