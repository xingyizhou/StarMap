import sys
import torch
from opts import opts
import ref
from utils.debugger import Debugger
import cv2
import numpy as np
from utils.img import Crop
from utils.hmParser import parseHeatmap
from utils.horn87 import horn87

def main():
  opt = opts().parse()
  model = torch.load(opt.loadModel)
  img = cv2.imread(opt.demo)
  s = max(img.shape[0], img.shape[1]) * 1.0
  c = np.array([img.shape[1] / 2., img.shape[0] / 2.])
  img = Crop(img, c, s, 0, ref.inputRes) / 256.
  input = torch.from_numpy(img.copy()).float()
  input = input.view(1, input.size(0), input.size(1), input.size(2))
  input_var = torch.autograd.Variable(input).float()
  if opt.GPU > -1:
    model = model.cuda(opt.GPU)
    input_var = input_var.cuda(opt.GPU)
  
  output = model(input_var)
  hm = output[-1].data.cpu().numpy()
  
  debugger = Debugger()
  img = (input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8).copy()
  inp = img.copy()
  star = (cv2.resize(hm[0, 0], (ref.inputRes, ref.inputRes)) * 255)
  star[star > 255] = 255
  star[star < 0] = 0
  star = np.tile(star, (3, 1, 1)).transpose(1, 2, 0)
  trans = 0.8
  star = (trans * star + (1. - trans) * img).astype(np.uint8)

   
  ps = parseHeatmap(hm[0], thresh = 0.1)
  canonical, pred, color, score = [], [], [], []
  for k in range(len(ps[0])):
    x, y, z = ((hm[0, 1:4, ps[0][k], ps[1][k]] + 0.5) * ref.outputRes).astype(np.int32)
    dep = ((hm[0, 4, ps[0][k], ps[1][k]] + 0.5) * ref.outputRes).astype(np.int32)
    canonical.append([x, y, z])
    pred.append([ps[1][k], ref.outputRes - dep, ref.outputRes - ps[0][k]])
    score.append(hm[0, 0, ps[0][k], ps[1][k]])
    color.append((1.0 * x / ref.outputRes, 1.0 * y / ref.outputRes, 1.0 * z / ref.outputRes))
    cv2.circle(img, (ps[1][k] * 4, ps[0][k] * 4), 4, (255, 255, 255), -1)
    cv2.circle(img, (ps[1][k] * 4, ps[0][k] * 4), 2, (int(z * 4), int(y * 4), int(x * 4)), -1)
  
  pred = np.array(pred).astype(np.float32)
  canonical = np.array(canonical).astype(np.float32)
  
  pointS = canonical * 1.0 / ref.outputRes
  pointT = pred * 1.0 / ref.outputRes
  R, t, s = horn87(pointS.transpose(), pointT.transpose(), score)
  
  rotated_pred = s * np.dot(R, canonical.transpose()).transpose() + t * ref.outputRes

  debugger.addImg(inp, 'inp')
  debugger.addImg(star, 'star')
  debugger.addImg(img, 'nms')
  debugger.addPoint3D(canonical / ref.outputRes - 0.5, c = color, marker = '^')
  debugger.addPoint3D(pred / ref.outputRes - 0.5, c = color, marker = 'x')
  debugger.addPoint3D(rotated_pred / ref.outputRes - 0.5, c = color, marker = '*')

  debugger.showAllImg(pause = True)
  debugger.show3D()

if __name__ == '__main__':
  main()
