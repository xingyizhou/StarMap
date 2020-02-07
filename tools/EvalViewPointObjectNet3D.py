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
from datasets.ObjectNet3D import ObjectNet3D
from utils.debugger import Debugger
from utils.hmParser import parseHeatmap
from utils.horn87 import RotMat, horn87
from scipy.linalg import logm
from progress.bar import Bar

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
acc30 = {}
acc10 = {}
err = {}
Acc10, Acc30 = 0, 0
for k, v in list(ref.ObjectNet3DClassName.items()):
  acc10[v], acc30[v], num[v] = 0, 0, 0
  err[v] = []
dataset = ObjectNet3D(opt, 'val')
bar = Bar('Eval', max = n)
for idx in range(n):
  index = idx if not DEBUG else np.random.randint(n)
  class_id = dataset.annot['class_id'][index]
  class_name = ref.ObjectNet3DClassName[class_id]
  
  v = np.array([dataset.annot['viewpoint_azimuth'][index], dataset.annot['viewpoint_elevation'][index], 
       dataset.annot['viewpoint_theta'][index]]) / 180.
  
  valid = dataset.annot['valid'][index]
  gt_model = np.array(dataset.annot['space_embedding'][index])[valid > 0]
  gt_view = v * PI
  anchors = np.array(dataset.annot['anchors_3d'][index][valid > 0])
  gt_point = Rotate(gt_model, gt_view)
  
  hm = preds[index]['map']
  ps = parseHeatmap(hm[0], thresh = 0.05)
  if len(ps[0]) == 0:
    num[class_name] += 1
    continue
  
  canonical = []
  pred = []
  color = []
  score = []
  for k in range(len(ps[0])):
    x, y, z = ((hm[0, 1:4, ps[0][k], ps[1][k]] + 0.5) * ref.outputRes).astype(np.int32)
    dep = ((hm[0, 4, ps[0][k], ps[1][k]] + 0.5) * ref.outputRes).astype(np.int32)
    score.append(hm[0, 0, ps[0][k], ps[1][k]])
    canonical.append([x, y, z])
    pred.append([ps[1][k], ref.outputRes - dep, ref.outputRes - ps[0][k]])
    
  pred = np.array(pred)
  canonical = np.array(canonical)
  score = np.array(score)
  
  Rs = []
  bestR = None
  minLoss = 1e10
  bestInliers = np.arange(pred.shape[0])
  
  
  ids = np.arange(pred.shape[0])
  gain = -1000
  pointT = np.array(pred[ids]) * 1.0 / ref.outputRes
  pointS = np.array(canonical[ids]) * 1.0 / ref.outputRes
  pointT[:, 2], pointT[:, 1] = - pointT[:, 1].copy(), pointT[:, 2].copy()
  R, t, s = horn87(pointS.transpose(), pointT.transpose(), score[ids])
  residual = ((np.dot(R, pointT.transpose()).transpose() + t - pointS) ** 2).sum(axis = 1).mean()
  bestR = R.copy()
  canonical_tmp = canonical.copy() * 1.0 / ref.outputRes
  pred_tmp = s * np.dot(R, canonical_tmp.transpose()).transpose() + t
  dists = ((pred_tmp - pred * 1.0 / ref.outputRes) ** 2).sum(axis = 1) ** 0.5

  R_gt = angle2dcm(gt_view)
  err_ = ((logm(np.dot(np.transpose(bestR), R_gt)) ** 2).sum()) ** 0.5 / (2.**0.5) * 180 / PI

  num[class_name] += 1
  acc30[class_name] += 1 if err_ <= 30. else 0
  acc10[class_name] += 1 if err_ <= 10. else 0
  err[class_name].append(err_)
  Acc30 += 1 if err_ <= 30. else 0
  Acc10 += 1 if err_ <= 10. else 0
  bar.suffix = '[{0}/{1}]|Total: {total:} | ETA: {eta:} | Acc_10: {Acc10:.6f} | Acc_30: {Acc30:.6f}'.format(idx, n, total = bar.elapsed_td, eta = bar.eta_td, Acc10 = Acc10 / (idx + 1.), Acc30 = Acc30 / (idx + 1.)) 
  next(bar)
  if DEBUG:
    debugger = Debugger()
    input, target, mask = dataset[index]
    img = (input[:3].transpose(1, 2, 0)*256).astype(np.uint8).copy()
    star = (cv2.resize(hm[0, 0], (ref.inputRes, ref.inputRes)) * 255)
    star[star > 255] = 255
    star[star < 0] = 0
    star = star.astype(np.uint8)

    for k in range(len(ps[0])):
      x, y, z = ((hm[0, 1:4, ps[0][k], ps[1][k]] + 0.5) * ref.outputRes).astype(np.int32)
      dep = ((hm[0, 4, ps[0][k], ps[1][k]] + 0.5) * ref.outputRes).astype(np.int32)
      color.append((1.0 * x / ref.outputRes, 1.0 * y / ref.outputRes, 1.0 * z / ref.outputRes))
      cv2.circle(img, (ps[1][k] * 4, ps[0][k] * 4), 6, (int(x * 4), int(y * 4), int(z * 4)), -1)
    debugger.addImg(img)
    debugger.addImg(star, 'star')
    debugger.addPoint3D(np.array((pred)) / 64. - 0.5, c = color, marker = 'x')
    rotated = Rotate(canonical, gt_view)
    rotated[:, 2], rotated[:, 1] = - rotated[:, 1].copy(), - rotated[:, 2].copy()
    debugger.addPoint3D(np.array(rotated) / 64. - 0.5, c = color, marker = '^')
    
    debugger.showAllImg(pause = False)
    debugger.show3D()
bar.finish()
accAll10 = 0.
accAll30 = 0.
numAll = 0.
mid = {}
err_all = []
for k, v in list(ref.ObjectNet3DClassName.items()):
  accAll10 += acc10[v]
  accAll30 += acc30[v]
  numAll += num[v]
  if num[v] > 0:
    acc10[v] = 1.0 * acc10[v] / num[v]
    acc30[v] = 1.0 * acc30[v] / num[v]
    mid[v] = np.sort(np.array(err[v]))[len(err[v]) // 2] 
  else:
    acc10[v] = 0
    acc30[v] = 0
    mid[v] = 0
  err_all = err_all + err[v]
print('Acc10', acc10)
print('Acc30', acc30)
print('mid', mid)
print('acc10All', accAll10 / numAll)
print('acc30All', accAll30 / numAll)
print('midAll', np.sort(np.array(err_all))[len(err_all) // 2])

