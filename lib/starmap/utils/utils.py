from __future__ import division
from builtins import object
from past.utils import old_div
from numpy.random import randn
import numpy as np

from .. import ref

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = old_div(self.sum, self.count)
             
def Rnd(x):
  return max(-2 * x, min(2 * x, randn() * x))
  
def Flip(img):
  if len(img.shape) == 3:
    return img[:, :, ::-1].copy()  
  elif len(img.shape) == 4:
    return img[:, :, :, ::-1].copy()  
  else:
    raise Exception('Flip shape error')
  
