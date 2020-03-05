from builtins import zip
from builtins import range
from builtins import object
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D

from .. import ref

def show2D(img, points, c):
  points = ((points.reshape(ref.nJoints, -1))).astype(np.int32)
  for j in range(ref.nJoints):
    cv2.circle(img, (points[j, 0], points[j, 1]), 3, c, -1)
  
  return img

class Debugger(object):
  def __init__(self):
    self.plt = plt
    self.fig = self.plt.figure()
    self.ax = self.fig.add_subplot((111),projection='3d')
    self.ax.grid(False)
    oo = 1e10
    self.xmax, self.ymax, self.zmax = -oo, -oo, -oo
    self.xmin, self.ymin, self.zmin = oo, oo, oo
    self.imgs = {}
    self.vols = {}
  
  def addPoint3D(self, points, c = 'b', marker = 'o', edges = []):
    points = points.reshape(-1, 3)
    x, y, z = np.zeros((3, points.shape[0]))
    for j in range(points.shape[0]):
      x[j] = points[j, 0].copy()
      y[j] = points[j, 1].copy()
      z[j] = points[j, 2].copy()
      self.xmax = max(x[j], self.xmax)
      self.ymax = max(y[j], self.ymax)
      self.zmax = max(z[j], self.zmax)
      self.xmin = min(x[j], self.xmin)
      self.ymin = min(y[j], self.ymin)
      self.zmin = min(z[j], self.zmin)
    if c == 'auto':
      c = [(z[j] + 0.5, y[j] + 0.5, x[j] + 0.5) for j in range(points.shape[0])]
    #print 'shape', x.shape, y.shape, z.shape, points.shape, c
    self.ax.scatter(x, y, z, s = 200, c = c, marker = marker)
    for e in edges:
      self.ax.plot(x[e], y[e], z[e], c = c)
    
  def show3D(self):
    max_range = np.array([self.xmax-self.xmin, self.ymax-self.ymin, self.zmax-self.zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(self.xmax+self.xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(self.ymax+self.ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(self.zmax+self.zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
      self.ax.plot([xb], [yb], [zb], 'w')
    self.plt.show()
    
  def addImg(self, img, imgId = 'default'):
    self.imgs[imgId] = img.copy()
  
  def addPoint2D(self, point, c, imgId = 'default'):
    self.imgs[imgId] = show2D(self.imgs[imgId], point, c)
  
  def showImg(self, pause = False, imgId = 'default'):
    cv2.imshow('{}'.format(imgId), self.imgs[imgId])
    if pause:
      cv2.waitKey()
  
  def showAllImg(self, pause = False):
    for i, v in list(self.imgs.items()):
      cv2.imshow('{}'.format(i), v)
    if pause:
      cv2.waitKey()
  
  def save3D(self, path):
    max_range = np.array([self.xmax-self.xmin, self.ymax-self.ymin, self.zmax-self.zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(self.xmax+self.xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(self.ymax+self.ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(self.zmax+self.zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
      self.ax.plot([xb], [yb], [zb], 'w')
    self.plt.savefig(path, bbox_inches='tight', frameon = False)
  
  def saveImg(self, imgId = 'default', path = '../debug/'):
    cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])
    
  def saveAllImg(self, path = '../debug/'):
    for i, v in list(self.imgs.items()):
      cv2.imwrite(path + '/{}.png'.format(i), v)
    
