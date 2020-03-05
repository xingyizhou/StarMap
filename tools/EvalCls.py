from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import range
import _init_paths
import sys
import numpy as np
import cv2
from starmap import ref
import torch
from datasets.Pascal3D import Pascal3D
from utils.debugger import Debugger
from utils.hmParser import parseHeatmap
from utils.horn87 import RotMat, horn87
from scipy.linalg import logm
PI = np.arccos(-1)
DEBUG = False


def angle2dcm(angle):
  azimuth = angle[0]
  elevation = angle[1]
  theta = angle[2]
  return np.dot(RotMat('Z', theta), np.dot(RotMat('X', - (PI / 2 - elevation)), RotMat('Z', - azimuth)))
  
def Rotate(points, angle):
  azimuth = angle[0]
  elevation = angle[1]
  theta = angle[2]
  azimuth = - azimuth
  elevation = - (PI / 2 - elevation)
  Ra = RotMat('Z', azimuth)
  Re = RotMat('X', elevation)
  Rt = RotMat('Z', theta)
  ret = np.dot(np.dot(Rt, np.dot(Re, Ra)), points.transpose()).transpose()
  ret[:, 1] *= -1
  return ret
  

path = sys.argv[1]
preds = torch.load(path)
opt = preds['opt']
preds = preds['preds']
n = len(preds)

num = {}
acc = {}
err = {}
for k, v in list(ref.pascalClassName.items()):
  acc[v], num[v] = 0, 0
  err[v] = []
  
dataset = Pascal3D(opt, 'val')
print('Evaluating...')
for idx in range(n):
  index = idx if not DEBUG else np.random.randint(n)
  
  class_id = dataset.annot['class_id'][index]
  class_name = ref.pascalClassName[class_id]
  v = np.array([dataset.annot['viewpoint_azimuth'][index], dataset.annot['viewpoint_elevation'][index], dataset.annot['viewpoint_theta'][index]]) / 180.
  gt_view = v * PI
  output = preds[index]['reg']
  numBins = opt.numBins
  binSize = 360. / opt.numBins
  
  try:
    _, pred = torch.from_numpy(output).view(3, numBins).topk(1, 1, True, True)
  except:
    _, pred = torch.from_numpy(output[0])[class_id * 3 * numBins:(class_id + 1) * 3 * numBins].view(3, numBins).topk(1, 1, True, True)
  #https://github.com/shubhtuls/ViewpointsAndKeypoints/blob/10fe7c7a74b3369dce9a3a13b3a7f85af859435b/utils/poseHypotheses.m#L53
  pred = (pred.view(3).float()).numpy()
  pred[0] = (pred[0] + 0.5) * PI / (opt.numBins / 2.)
  pred[1] = (pred[1] - opt.numBins / 2) * PI / (opt.numBins / 2.)
  pred[2] = (pred[2] - opt.numBins / 2) * PI / (opt.numBins / 2.)

  bestR = angle2dcm(pred)
  
  R_gt = angle2dcm(gt_view)
  err_ = ((logm(np.dot(np.transpose(bestR), R_gt)) ** 2).sum()) ** 0.5 / (2.**0.5) * 180 / PI

  num[class_name] += 1
  acc[class_name] += 1 if err_ <= 30. else 0
  err[class_name].append(err_)

  if DEBUG:
    input, target, mask, view = dataset[index]
    debugger = Debugger()
    img = (input[:3].transpose(1, 2, 0)*256).astype(np.uint8).copy()
    debugger.addImg(img)
    debugger.showAllImg(pause = False)


accAll = 0.
numAll = 0.
mid = {}
err_all = []
for k, v in list(ref.pascalClassName.items()):
  accAll += acc[v]
  numAll += num[v]
  acc[v] = 1.0 * acc[v] / num[v]
  mid[v] = np.sort(np.array(err[v]))[len(err[v]) // 2] 
  err_all = err_all + err[v]
print('Acc', acc)
print('num', num)
print('mid', mid)
print('accAll', accAll / numAll)
print('midAll', np.sort(np.array(err_all))[len(err_all) // 2]) 




