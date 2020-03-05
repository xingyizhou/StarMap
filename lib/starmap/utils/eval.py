from __future__ import division
from builtins import range
from past.utils import old_div
import numpy as np
from scipy.linalg import logm

from .. import ref

def getPreds(hm):
  assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
  res = hm.shape[2]
  hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
  idx = np.argmax(hm, axis = 2)
  preds = np.zeros((hm.shape[0], hm.shape[1], 2))
  for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
      preds[i, j, 0], preds[i, j, 1] = idx[i, j] % res, old_div(idx[i, j], res)
  
  return preds

def calcDists(preds, gt, normalize):
  dists = np.zeros((preds.shape[1], preds.shape[0]))
  for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
      if gt[i, j, 0] > 0 and gt[i, j, 1] > 0:
        dists[j][i] = old_div(((gt[i][j] - preds[i][j]) ** 2).sum() ** 0.5, normalize[i])
      else:
        dists[j][i] = -1
  return dists

def distAccuracy(dist, thr = 0.5):
  dist = dist[dist != -1]
  if len(dist) > 0:
    return 1.0 * (dist < thr).sum() / len(dist)
  else:
    return -1

  
def AccCls(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res


sqrt2 = 2.0 ** 0.5
pi = np.arccos(-1)
'''
    % First roation along Z by azimuth. Then along X by -(pi/2-elevation).
    % Then along Z by theta
'''

def RotMat(axis, ang):
  s = np.sin(ang)
  c = np.cos(ang)
  res = np.zeros((3, 3))
  if axis == 'Z':
    res[0, 0] = c
    res[0, 1] = -s
    res[1, 0] = s
    res[1, 1] = c
    res[2, 2] = 1
  elif axis == 'Y':
    res[0, 0] = c
    res[0, 2] = s
    res[1, 1] = 1
    res[2, 0] = -s
    res[2, 2] = c
  elif axis == 'X':
    res[0, 0] = 1
    res[1, 1] = c
    res[1, 2] = -s
    res[2, 1] = s
    res[2, 2] = c
  return res

def angle2dcm(angle):
  azimuth = angle[0]
  elevation = angle[1]
  theta = angle[2]
  return np.dot(RotMat('Z', theta), np.dot(RotMat('X', - (old_div(pi, 2) - elevation)), RotMat('Z', - azimuth)))

def AccViewCls(output, target, numBins, specificView): 
  #unified
  binSize = 360. / numBins
  if specificView:
    acc = 0
    for t in range(target.shape[0]):
      idx = np.where(target[t] != numBins)
      ps = old_div(idx[0][0], 3 * 3)
      _, pred = output[t].view(-1, numBins)[ps: ps + 3].topk(1, 1, True, True)
      pred = pred.view(3).float() * binSize / 180. * pi 
      gt = target[t][ps: ps + 3].float() * binSize / 180. * pi
      R_pred = angle2dcm(pred)
      R_gt = angle2dcm(gt)
      err = old_div(((logm(np.dot(np.transpose(R_pred), R_gt)) ** 2).sum()) ** 0.5, sqrt2)
      acc += 1 if err < pi / 6. else 0
    return 1.0 * acc / target.shape[0]
  else:
    _, pred = output.view(target.shape[0] * 3, numBins).topk(1, 1, True, True)
    pred = pred.view(target.shape[0], 3).float() * binSize / 180. * pi
    target = target.float() * binSize / 180. * pi
    acc = 0
    for t in range(target.shape[0]):
      R_pred = angle2dcm(pred[t])
      R_gt = angle2dcm(target[t])
      err = old_div(((logm(np.dot(np.transpose(R_pred), R_gt)) ** 2).sum()) ** 0.5, sqrt2)
      acc += 1 if err < pi / 6. else 0
    return 1.0 * acc / target.shape[0]


