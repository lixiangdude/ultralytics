import numpy as np
import torch

from ultralytics.nn.modules.block import SELayer, SPPFSE, MPGD, APGD

if __name__ == '__main__':
    a = torch.ones(3, 16, 1024, 1024)
    # print(a)
    # b = torch.Tensor([1, 2, 3]).reshape((3, 1))
    # print(b)
    #
    # print(torch.mul(a, b))
    res = APGD(512, 1024)(a)
    print(res)
