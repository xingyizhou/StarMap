from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import ref 
import os
import scipy.io as sio
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D


oo = 0.5
PI = np.arccos(-1)
DEBUG = False
SIZE = 224
PATH = ref.pascal3dDir
MODEL_PATH = PATH + 'CAD/'
ANNOT_PATH = PATH + 'Annotations/'
IMAGE_PATH = PATH + 'Images/'
IMGNET_SPLIT_PATH = ref.pascal3dDir + 'Image_sets/'
PASCAL_SPLIT_PATH = ref.pascal3dDir + 'PASCAL/VOCdevkit/VOC2012/ImageSets/Main/'

pascalPointInfo = ref.pascalPointInfo
maxNKeypoints = max([len(pascalPointInfo[v]) for v in pascalPointInfo])
maxLenName = max([len(v) for v in pascalPointInfo])
classId = {v: i for i, v in enumerate(pascalPointInfo)}
#classId = {'train'}
color = ['r', 'g', 'lime', 'k', 'y', 'm', 'c', 'tan', 'pink', 'yellow', 'b', 'navy']
'''
'diningtable':
   ['leg_upper_left',
    'leg_upper_right',
    'leg_lower_left',
    'leg_lower_right',
    'top_upper_left',
    'top_upper_right',
    'top_lower_left',
    'top_lower_right',
    'top_up',
    'top_down',
    'top_left',
    'top_right'],
'''
color_motor = ['k', 'k', 'k', 'k', 'g', 'g', 'g', 'b', 'b', 'b']
color_table = ['g', 'b', 'g', 'b', 'g', 'b', 'g', 'b', 'k', 'k', 'g', 'b']

x, y, z = {}, {}, {}
val = {}
for cls in classId:
  x[cls], y[cls], z[cls], val[cls] = [], [], [], []
  model_path = MODEL_PATH + '{}.mat'.format(cls)
  models = sio.loadmat(model_path)[cls][0]
  model_pts = np.zeros((len(models), maxNKeypoints, 3))
  valid = np.zeros((len(models), maxNKeypoints))
  for i in range(len(models)):
    for j, part in enumerate(ref.pascalPointInfo[cls]):
      try:
        model_pts[i, j] = models[i][part][0].copy()
        valid[i, j] = 1
        
      except:
        pass
    x[cls].append(model_pts[i, :, 0])
    y[cls].append(model_pts[i, :, 1])
    z[cls].append(model_pts[i, :, 2])
    val[cls].append(valid[i])
  x[cls] = np.array(x[cls])
  y[cls] = np.array(y[cls])
  z[cls] = np.array(z[cls])
  val[cls] = np.array(val[cls])
  
  #print model_pts
  
plt = plt
fig =  plt.figure()
ax =  fig.add_subplot((111),projection='3d')
ax.set_xlabel('x') 
ax.set_ylabel('y') 
ax.set_zlabel('z')
xmax,  ymax,  zmax = oo, oo, oo
xmin,  ymin,  zmin = -oo, -oo, -oo  
max_range = np.array([ xmax- xmin,  ymax- ymin,  zmax- zmin]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*( xmax+ xmin)
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*( ymax+ ymin)
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*( zmax+ zmin)
for xb, yb, zb in zip(Xb, Yb, Zb):
  ax.plot([xb], [yb], [zb], 'w')
  
for i, cls in enumerate(classId):
  c = [(z[cls].reshape(x[cls].size)[j] + 0.5, y[cls].reshape(x[cls].size)[j] + 0.5, x[cls].reshape(x[cls].size)[j] + 0.5) for j in range(len(x[cls].reshape(x[cls].size)))]
  ax.scatter(x[cls].reshape(x[cls].size), y[cls].reshape(x[cls].size), z[cls].reshape(x[cls].size), c = color[i], s = 10)
  #for j in range(len(x[cls])):
  #  for k in range(12):
  #    ax.scatter(x[cls][j][k], y[cls][j][k], z[cls][j][k], c = color_table[k])

  xx = x[cls].mean(axis = 0)
  yy = y[cls].mean(axis = 0)
  zz = z[cls].mean(axis = 0)
  for e in ref.pascalEdgeInfo[cls]:
    u, v = ref.pascalNamePointId[cls][e[0]], ref.pascalNamePointId[cls][e[1]]
    ax.plot(xx[[u, v]], yy[[u, v]], zz[[u, v]], c = color[i], linewidth=1)
    #for j in range(len(x[cls])):
    #  ax.plot(x[cls][j][[u, v]], y[cls][j][[u, v]], z[cls][j][[u, v]], c = color[i])

plt.savefig('embPascal.pdf', bbox_inches='tight')
plt.show()
  
  
  
