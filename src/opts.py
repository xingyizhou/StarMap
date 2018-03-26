import argparse
import os
import sys
import ref

class opts():
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('-GPU', type = int, default = -1, help = 'GPU id, -1 for CPU')
    self.parser.add_argument('-demo', default = '', help = 'path/to/demo/image')
    self.parser.add_argument('-loadModel', default = '../models/Pascal3D-cpu.pth', help = 'path/to/model')

  def parse(self):
    opt = self.parser.parse_args()

    return opt
