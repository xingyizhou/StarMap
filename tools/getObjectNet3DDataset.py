from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import scipy.io as sio
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
import ref
PI = np.arccos(-1)
oo = 1
DEBUG = False
SIZE = 224
PATH = '/home/zxy/Datasets/ObjectNet3D/'
MODEL_PATH = PATH + 'CAD/mat/'
ANNOT_PATH = PATH + 'Annotations/'
IMAGE_PATH = PATH + 'Images/'
SAVE_PATH = '/home/zxy/Projects/CrossObjectKeypoint/data/ObjectNet3D'
SPLIT_PATH = '/home/zxy/Datasets/ObjectNet3D/Image_sets/'
if not os.path.exists(SAVE_PATH):
  os.mkdir(SAVE_PATH)
tags = ['class', 'bbox', 'anchors', 'viewpoint', 'cad_index', 'truncated', 'occluded', 'difficult']
direct_tags = ['bbox', 'truncated', 'occluded', 'difficult']
tags_viewpoint = ['azimuth_coarse', 'elevation_coarse', 'azimuth', 'elevation', 'distance',
         'focal', 'px', 'py', 'theta', 'error', 'interval_azimuth', 'interval_elevation', 
         'num_anchor', 'viewport']

classNames = [v for v in sorted(ref.ObjectNet3DPointInfo)]

print classNames

val_classes = []
for i, cls in enumerate(classNames):
  if i % 5 == 4:
    val_classes.append(cls)


print('val classes', val_classes)

classId = {v: i for i, v in enumerate(classNames)}

maxNKeypoints = 24
maxLenName = 17

model_pts = {}
valid = {}
PointInfo = {}

numModels = 0
numMat = 0
numH5 = 0
for cls in classNames:
  model_path = MODEL_PATH + '{}.mat'.format(cls)
  PointInfo[cls] = []
  maxLenName = max(maxLenName, len(cls))
  matFormat = True
  try:
    models = sio.loadmat(model_path)[cls][0]
    numMat += 1
  except:
    models = h5py.File(model_path, 'r')[cls]
    matFormat = False
    numH5 += 1
  print cls, cls in val_classes
  if matFormat:
    model_pts[cls] = np.zeros((len(models), maxNKeypoints, 3))
    valid[cls] = np.zeros((len(models), maxNKeypoints))
    for i in range(len(models)):
      for j, part in enumerate(models['pnames'][0][0]):
        if i == 0:
          PointInfo[cls].append(part[0])
        try:
          model_pts[cls][i, j] = models[i][part[0]][0].copy()
          valid[cls][i, j] = 1
        except:
          pass
  else:
    model_pts[cls] = np.zeros((len(models['pnames']), maxNKeypoints, 3))
    valid[cls] = np.zeros((len(models['pnames']), maxNKeypoints))
    for i in range(len(models['pnames'])):
      for j, part in enumerate(models[models['pnames'][i][0]]):
        str1 = ''.join(chr(k) for k in models[part[0]][:])
        if i == 0:
          PointInfo[cls].append(str1)
        try:
          model_pts[cls][i, j] = np.array(models[models[str1][i][0]]).reshape(3)
          valid[cls][i, j] = 1
        except:
          pass

#maxNKeypoints = max([len(pascalPointInfo[v]) for v in pascalPointInfo])
#maxLenName = max([len(v) for v in pascalPointInfo])
#print 'maxLenName', maxLenName
maxImgName = -1 

for split in ['train', 'val']:
  nProjectedAnchor = 0
  total = 0
  annot = {}
  keys = tags + ['viewpoint_' + v for v in tags_viewpoint] + \
       ['status', 'vis', 'dataset', 'class_id', 'imgname', 'projected_anchors', 'anchors_3d', 'valid', 'space_embedding']

  for tag in keys:
    annot[tag] = []

  split_file = SPLIT_PATH + '/{}.txt'.format(split)
  f = open(split_file, 'r')
  files = [line[:-1] for line in f]
  cnt = 0
  numNoParts = 0
  numNoAnchors = 0
  numLeave = 0
  numHalfAnchors = 0
  if 1:  
    img_path = IMAGE_PATH
    for (t, im_name) in enumerate(files):
      im_ = im_name

      annot_file = ANNOT_PATH + im_name
      try:
        data_ = sio.loadmat(annot_file)
      except:
        print('MISS DATA', im_name)
        continue
      data_objects = data_['record']['objects'][0][0]

      data = {}
      for tag in tags:
        data[tag] = data_objects[tag][0]

      for i in range(data['anchors'].shape[0]):
        total += 1
        hasData = True
        if len(data['viewpoint'][i]) > 0:
          try:
            anchors = data['anchors'][i][0]
          except:
            anchors = None
          viewpoint = data['viewpoint'][i][0]
        else:
          numNoAnchors += 1
          if len(data['viewpoint'][i]) > 0:
            numHalfAnchors += 1
          hasData = False
        #if data['difficult'][i][0] == 1 or data['truncated'][i][0] == 1 or data['occluded'][i][0] == 1:
        #  continue
        
        if data['difficult'][i][0] > 0:
          continue
        
        if hasData and len(viewpoint) > 0:
          class_name = data['class'][i][0]
          cls = class_name
          maxImgName = max(maxImgName, len(im_name))
          anchor = np.zeros((maxNKeypoints, 2))
          status = np.zeros((maxNKeypoints))
          vis = np.zeros((maxNKeypoints))
          numParts = 0
          for (j, point_name) in enumerate(PointInfo[class_name]):
            hasPart = True
            try:
              if len(anchors[point_name][0]) > 0:
                part = anchors[point_name][0][0][0]
                if len(part['status']) > 0:            
                  status[j] = part['status'][0][0]
              else:
                hasPart = False
            except:
              hasPart = False
            if hasPart and len(part['location']) > 0:
              anchor[j] = part['location'][0]
              vis[j] = 1
              numParts += 1
          if numParts == 0:
            numNoParts += 1
            continue

          ProjectPoint = False

          if len(data['cad_index'][i][0]) > 0:
            if 1:
              viewpoint = data['viewpoint'][i][0]
              try:
                try:
                  a = viewpoint['azimuth'][0][0][0] * PI / 180.
                  e = viewpoint['elevation'][0][0][0] * PI / 180.
                except:
                  a = viewpoint['azimuth_coarse'][0][0][0] * PI / 180.
                  e = viewpoint['elevation_coarse'][0][0][0] * PI / 180.
                d = viewpoint['distance'][0][0][0]
                f = viewpoint['focal'][0][0][0]
                theta = viewpoint['theta'][0][0][0] * PI / 180.
                principal = np.array([viewpoint['px'][0][0][0], viewpoint['py'][0][0][0]])
                viewport = viewpoint['viewport'][0][0][0]
                if a == 0 and e == 0 and theta == 0:
                  continue
              except:
                print 'NO ANNOTATION'
                continue
              C = np.array([d * np.cos(e) * np.sin(a), -d * np.cos(e) * np.cos(a), d * np.sin(e)]).reshape(3, 1)
              
              a = - a
              e = - (PI / 2 - e)
              
              Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
              Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
              R = np.dot(Rx, Rz)
              
              M = viewport
              #P = [M*f 0 0; 0 M*f 0; 0 0 -1] * [R -R*C];
              P_ = np.array([[M * f, 0, 0], [0, M * f, 0], [0, 0, -1]])
              #print 'shape R C', R.shape, C.shape
              P = np.dot(P_, np.concatenate([R, - np.dot(R, C)], axis = 1))
              

              #print 'shape P', P.shape 
              x3d = model_pts[cls][int(data['cad_index'][i][0]) - 1]
              x3d = np.dot(P, np.concatenate([x3d, np.ones((x3d.shape[0], 1))], axis = 1).transpose())
              #x3d = np.dot(R, x3d.transpose())
              x = x3d.copy()
              #print 'x3d __ 0', x3d, x3d.shape
              x[2, x[2, :] == 0] = 1
              x[0] = x[0] / x[2]
              x[1] = x[1] / x[2]
              x = x[:2]
              
              R2d = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
              x = np.dot(R2d, x).transpose()
              x[:, 1] = - x[:, 1]
              x[:, 0] += principal[0]
              x[:, 1] += principal[1]
              
              p3d = np.dot(R, model_pts[cls][int(data['cad_index'][i][0]) - 1].transpose()).transpose()
              p3d[:, :2] = np.dot(R2d, p3d[:, :2].transpose()).transpose()
              p3d[:, 1] = - p3d[:, 1]
              mean_p = p3d.mean(axis = 0)
              std_p = max(p3d[:, 0].max() - p3d[:, 0].min(), p3d[:, 1].max() - p3d[:, 1].min())
              mean_x = x.mean(axis = 0)
              std_x = max(x[:, 0].max() - x[:, 0].min(), x[:, 1].max() - x[:, 1].min())
              for j in range(p3d.shape[0]):
               p3d[j, 0] = (p3d[j, 0] - mean_p[0]) / std_p * std_x + mean_x[0]
               p3d[j, 1] = (p3d[j, 1] - mean_p[1]) / std_p * std_x + mean_x[1]
               p3d[j, 2] = (p3d[j, 2] - mean_p[2]) / std_p * std_x
             
              ProjectPoint = True
              
              if DEBUG:
                
                fig = plt.figure()
                ax = fig.add_subplot((111),projection='3d')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                xmax, ymax, zmax = oo, oo, oo
                xmin, ymin, zmin = -oo, -oo, -oo
                
                nJoints = len(PointInfo[class_name])
                #points = model_pts[cls][int(data['cad_index'][i][0]) - 1].reshape(nJoints, 3)
                points = p3d#.transpose()
                points = points - points.mean(axis = 0)
                #print 'points', points
                xx, yy, zz = np.zeros((3, nJoints))
                for j in range(nJoints):
                  xx[j] = points[j, 0] 
                  yy[j] = points[j, 1] 
                  zz[j] = points[j, 2]
                ax.scatter(xx, yy, zz, c = 'r')
                max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
                Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
                Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
                Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
                for xb, yb, zb in zip(Xb, Yb, Zb):
                  ax.plot([xb], [yb], [zb], 'w')
                
                img_path = '{}/{}.JPEG'.format(IMAGE_PATH, im_name)
                print img_path
                img = cv2.imread(img_path)
                bb = data['bbox'][i][0]
                
                for j in range(maxNKeypoints):
                  if vis[j] > 0.5:
                    cv2.circle(img, (int(anchor[j, 0]), int(anchor[j, 1])), 6, (0, 255, 255), -1) #Annotation
                  else:
                    cv2.circle(img, (int(anchor[j, 0]), int(anchor[j, 1])), 6, (255, 0, 255), -1) #Annotation
                for j in range(maxNKeypoints):
                  if valid[cls][int(data['cad_index'][i][0]) - 1][j] > 0.5:
                    cv2.circle(img, (int(x[j, 0]), int(x[j, 1])), 4, (0, 0, 255), -1) #Projection
                for j in range(maxNKeypoints):
                  if valid[cls][int(data['cad_index'][i][0]) - 1][j] > 0.5:
                    cv2.circle(img, (int(p3d[j, 0]), int(p3d[j, 1])), 2, tuple(np.array((255, 0, 0)) * 1.0 * j / nJoints), -1) #Affine 3D projection
                cv2.imshow('img', img)
                cv2.waitKey()
            else:
              print 'PROJECTION ERROR'
              pass
          else:
            print 'MISS CAD'
            continue
          
          if ProjectPoint:
            nProjectedAnchor += 1
            projected_anchor = np.zeros((maxNKeypoints, 2))
            anchors_3d = np.zeros((maxNKeypoints, 3))
            for j in range(x.shape[0]):
             projected_anchor[j] = x[j].copy()
             anchors_3d[j] = p3d[j].copy()
          else:
            print 'PROJECTION ERROR'
            continue
            
          cnt += 1
          
          for j in range(anchor.shape[0]):
            if valid[cls][int(data['cad_index'][i][0]) - 1][j] < 0.5:
              anchor[j] = anchor[j] * 0 - 1
              projected_anchor[j] = projected_anchor[j] * 0 - 1
              anchors_3d[j] = anchors_3d[j] * 0 - 1

          annot['projected_anchors'] += [projected_anchor]
          annot['anchors_3d'] += [anchors_3d]
          annot['cad_index'] += [int(data['cad_index'][i][0])]
          annot['valid'] += [valid[cls][int(data['cad_index'][i][0]) - 1]]
          annot['anchors'] += [anchor]
          annot['status'] += [status]
          annot['vis'] += [vis]
          annot['dataset'] += ['ObjectNet3D']
          annot['class_id'] += [classId[class_name]]
          annot['space_embedding'] += [model_pts[cls][int(data['cad_index'][i][0]) - 1]]
          
          class_ = np.zeros(maxLenName, dtype = np.uint8)
          for v in range(len(class_name)): 
            class_[v] = ord(class_name[v])
          annot['class'] += [class_]
          
          imgname_ = np.zeros(22, dtype = np.uint8)
          for v in range(len(im_name)): 
            imgname_[v] = ord(im_name[v])
          annot['imgname'] += [imgname_]
          
          #print class_
          for tag in direct_tags:
            annot[tag] += [data[tag][i][0]]
          for tag_viewpoint in tags_viewpoint:
            if 'coarse' in tag_viewpoint:
              annot['viewpoint_' + tag_viewpoint] += [data['viewpoint'][i][0][tag_viewpoint][0][0][0]]
            else:
              try:
                annot['viewpoint_' + tag_viewpoint] += [data['viewpoint'][i][0][tag_viewpoint][0][0][0]]
              except:
                if tag_viewpoint == 'azimuth' or tag_viewpoint == 'elevation':
                  annot['viewpoint_' + tag_viewpoint] += [data['viewpoint'][i][0][tag_viewpoint + '_coarse'][0][0][0]]
                else:
                  annot['viewpoint_' + tag_viewpoint] += [0]
    print 'cnt =', cnt, ',numNoParts =', numNoParts, ',numNoAnchors =', numNoAnchors, ', numLeave', numLeave, ',total =', total
    
  with h5py.File('{}/ObjectNet3D-{}.h5'.format(SAVE_PATH, split),'w') as f:
    f.attrs['name'] = 'ObjectNet3D-{}'.format(split)
    for k in keys:
      f[k] = np.array(annot[k])
