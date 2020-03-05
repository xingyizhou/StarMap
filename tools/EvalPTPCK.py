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
from utils.img import Transform
from scipy.linalg import logm
import scipy.io as sio
PI = np.arccos(-1)
DEBUG = False
GTASSIGN = False
GTCAD = False
PATH = ref.pascal3dDir
MODEL_PATH = PATH + 'CAD/'

print('Loading average model.')
avgModel = {}
model_pts = {}
pascalPointInfo = ref.pascalPointInfo
maxNKeypoints = max([len(pascalPointInfo[v]) for v in pascalPointInfo])
for k, cls in list(ref.pascalClassName.items()):
  model_path = MODEL_PATH + '{}.mat'.format(cls)
  models = sio.loadmat(model_path)[cls][0]
  model_pts[cls] = np.zeros((len(models), maxNKeypoints, 3))
  valid = np.zeros((len(models), maxNKeypoints))
  avgModel[cls] = np.zeros((maxNKeypoints, 3))
  cntAvg = np.zeros((maxNKeypoints))
  for i in range(len(models)):
    for j, part in enumerate(ref.pascalPointInfo[cls]):
      try:
        model_pts[cls][i, j] = models[i][part][0].copy()
        valid[i, j] = 1
        avgModel[cls][j] += models[i][part][0]
        cntAvg[j] += 1
      except:
        pass
  for j in range(maxNKeypoints):
    if cntAvg[j] > 0:
      avgModel[cls][j] /= cntAvg[j]

path = sys.argv[1]
preds = torch.load(path)
opt = preds['opt']
preds = preds['preds']
n = len(preds)
num = {}
acc = {}
err = {}
ptAccC = {}
cntPt = {}
for k, v in list(ref.pascalClassName.items()):
  acc[v], num[v] = 0, 0
  err[v] = []
  ptAccC[v], cntPt[v] = 0, 0
fewPoints = 0
dataset = Pascal3D(opt, 'val')
print('Evaluating...')
for idx in range(n):
  index = idx if not DEBUG else np.random.randint(n)
  hm = preds[index]['map']
  img = dataset.LoadImage(index)
  class_id = dataset.annot['class_id'][index]
  class_name = ref.pascalClassName[class_id]
  pts2d, pts3d, emb, c, s = dataset.GetPartInfo(index)
  s = min(s, max(img.shape[0], img.shape[1])) * 1.0
  r = 0.
  
  box = dataset.annot['bbox'][index].copy()
  thresh = max((box[2] - box[0]), (box[3] - box[1])) * 0.1
  if 1:
    ps = parseHeatmap(hm[0], thresh = 0.05)
   
    for i in range(pts2d.shape[0]):
      if dataset.annot['valid'][index][i] > ref.eps and dataset.annot['vis'][index][i] > ref.eps:
        cntPt[class_name] += 1.
        gtFeat = avgModel[class_name][i]
        
        gtLoc = Transform(pts2d[i], c, s, r, ref.outputRes).astype(np.int32)
        minDis = 1e10
        pt = np.zeros(2)
        for k in range(len(ps[0])):
          feat = hm[0, 1:4, ps[0][k], ps[1][k]]
          if GTASSIGN:
            dis = ((gtLoc[0] - ps[1][k]) ** 2 + (gtLoc[1] - ps[0][k]) ** 2)
          else:
            dis = ((feat - gtFeat) ** 2).sum()
          if dis < minDis:
            minDis = dis
            pt[0], pt[1] = ps[1][k], ps[0][k]
        pred = Transform(pt, c, s, r, ref.outputRes, invert = True)
        
        if ((pred - pts2d[i]) ** 2).sum() ** 0.5 < thresh:
          ptAccC[class_name] += 1.


accAll = 0.
numAll = 0.
mid = {}
err_all = []
for k, v in list(ref.pascalClassName.items()):
  accAll += ptAccC[v]
  numAll += cntPt[v]
  ptAccC[v] /= cntPt[v]
  #print(v, ptAccC[v])

print('|        |', end='')
for k in sorted(ptAccC):
  print('{:8s}|'.format(k), end='')
print(' Mean |')
for k in range(len(sorted(ptAccC)) + 2):
  print('|--------', end='')
print('|')
print('| Acc    |', end='')
for k in sorted(ptAccC):
  print('{:.4f}'.format(ptAccC[k]), ' |', end='')
print('{:.4f} |'.format(accAll / numAll))

#print('accAll', accAll / numAll)

