from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import zip
from builtins import range
import _init_paths

import starmap.ref  as ref
import os
import scipy.io as sio
import cv2
import numpy as np
import h5py
PI = np.arccos(-1)
DEBUG = False
SIZE = 224
PATH = ref.pascal3dDir
MODEL_PATH = PATH + 'CAD/'
ANNOT_PATH = PATH + 'Annotations/'
IMAGE_PATH = PATH + 'Images/'
SAVE_PATH = ref.dataDir + 'Pascal3D'
IMGNET_SPLIT_PATH = ref.pascal3dDir + 'Image_sets/'
PASCAL_SPLIT_PATH = ref.pascal3dDir + 'PASCAL/VOCdevkit/VOC2012/ImageSets/Main/'
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

objs = ['bottle_pascal', 'train_imagenet', 'bus_pascal', 'aeroplane_pascal', 'diningtable_imagenet', 'chair_imagenet', 'boat_imagenet', 'bicycle_imagenet', 'bicycle_pascal', 'tvmonitor_pascal',  'bus_imagenet', 'tvmonitor_imagenet', 'aeroplane_imagenet', 'chair_pascal', 'diningtable_pascal',  'train_pascal', 'bottle_imagenet', 'boat_pascal', 'motorbike_imagenet','motorbike_pascal', 'car_imagenet', 'car_pascal', 'sofa_imagenet', 'sofa_pascal']
tags = ['class', 'bbox', 'anchors', 'viewpoint', 'cad_index', 'truncated', 'occluded', 'difficult']
direct_tags = ['bbox', 'truncated', 'occluded', 'difficult']
tags_viewpoint = ['azimuth_coarse', 'elevation_coarse', 'azimuth', 'elevation', 'distance',
                  'focal', 'px', 'py', 'theta', 'error', 'interval_azimuth', 'interval_elevation', 
                  'num_anchor', 'viewport']
pascalPointInfo = ref.pascalPointInfo
pascalClasses = ref.pascalClasses
classId = {v: i for i, v in enumerate(pascalClasses)}

maxNKeypoints = max([len(pascalPointInfo[v]) for v in pascalPointInfo])
maxLenName = max([len(v) for v in pascalPointInfo])
print('maxNKeypoints', maxNKeypoints)
print('maxLenName', maxLenName)

for split in ['train', 'val']:
    nProjectedAnchor = 0
    total = 0
    annot = {}
    keys = tags + ['viewpoint_' + v for v in tags_viewpoint] + ['status', 'vis', 'dataset', 'class_id', 'imgname', 'projected_anchors', 'anchors_3d', 'valid', 'space_embedding', 'avg_embedding']
    for tag in keys:
      annot[tag] = []
    maxImgName = -1 

    for obj in objs:
        cls = obj[:obj.find('_')]
        model_path = MODEL_PATH + '{}.mat'.format(cls)
        models = sio.loadmat(model_path)[cls][0]
        model_pts = np.zeros((len(models), maxNKeypoints, 3))
        valid = np.zeros((len(models), maxNKeypoints))
        avgModel = np.zeros((maxNKeypoints, 3))
        avgCnt = np.zeros(maxNKeypoints)
        for i in range(len(models)):
            for j, part in enumerate(ref.pascalPointInfo[cls]):
                try:
                    model_pts[i, j] = models[i][part][0].copy()
                    valid[i, j] = 1
                    avgModel[j] += model_pts[i, j]
                    avgCnt[j] += 1
                except:
                    pass

        for j, part in enumerate(ref.pascalPointInfo[cls]):
          if avgCnt[j] > 0:
            avgModel[j] = avgModel[j] / avgCnt[j]
        if 'imagenet' in obj:
            if split == 'train':
                split_file = IMGNET_SPLIT_PATH + '/{}_imagenet_train.txt'.format(obj[:obj.find('_')])
                f = open(split_file, 'r')
                files = [line[:-1] for line in f]
                split_file = IMGNET_SPLIT_PATH + '/{}_imagenet_val.txt'.format(obj[:obj.find('_')])
                f = open(split_file, 'r')
                files += [line[:-1] for line in f]
            else:
                flies = []
        else:
            split_file = PASCAL_SPLIT_PATH + '/{}_{}.txt'.format(obj[:obj.find('_')], split)
            f = open(split_file, 'r')
            files = []
            for line in f:
              p, l = line[:-1].split(' ')[0], line[:-1].split(' ')[-1]
              if int(l) > 0:
                files.append(p)
        cnt = 0
        obj_path = SAVE_PATH + obj
        img_path = IMAGE_PATH + obj
        for (t, im_name) in enumerate(os.listdir(img_path)):
            im_ = im_name[:im_name.find('.')]
            if not (im_ in files):
              continue
            
            obj_id = im_name.split('.')[0]
            annot_file = ANNOT_PATH + obj + '/' + obj_id
            data_ = sio.loadmat(annot_file)
            data_objects = data_['record']['objects'][0][0]

            data = {}
            for tag in tags:
                data[tag] = data_objects[tag][0]

            for i in range(data['anchors'].shape[0]):
                anchors = data['anchors'][i][0]
                viewpoint = data['viewpoint'][i][0]
                
                if len(anchors) > 0 and len(viewpoint) > 0:
                    class_name = data['class'][i][0]
                    #https://github.com/shubhtuls/ViewpointsAndKeypoints/blob/10fe7c7a74b3369dce9a3a13b3a7f85af859435b/preprocess/readVpsDataPascal.m#L21
                    #if class_name != cls or data['difficult'][i][0] > 0:
                    #  continue
                    maxImgName = max(maxImgName, len(im_name))
                    anchor = np.zeros((maxNKeypoints, 2))
                    status = np.zeros((maxNKeypoints))
                    vis = np.zeros((maxNKeypoints))
                    for (j, point_name) in enumerate(pascalPointInfo[class_name]):
                        part = anchors[point_name][0][0][0]
                        status[j] = part['status'][0][0]
                        if len(part['location']) > 0:
                            anchor[j] = part['location'][0]
                            vis[j] = 1
                    ProjectPoint = False

                    if len(data['cad_index'][i][0]) > 0:
                        try:
                            viewpoint = data['viewpoint'][i][0]
                            a = viewpoint['azimuth'][0][0][0] * PI / 180.
                            e = viewpoint['elevation'][0][0][0] * PI / 180.
                            d = viewpoint['distance'][0][0][0]
                            f = viewpoint['focal'][0][0][0]
                            
                            theta = viewpoint['theta'][0][0][0] * PI / 180.
                            principal = np.array([viewpoint['px'][0][0][0], viewpoint['py'][0][0][0]])
                            viewport = viewpoint['viewport'][0][0][0]
                            if a == 0 and e == 0 and theta == 0:
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

                            P = np.dot(P_, np.concatenate([R, - np.dot(R, C)], axis = 1))
                            
                            x3d = model_pts[int(data['cad_index'][i][0]) - 1]
                            x3d = np.dot(P, np.concatenate([x3d, np.ones((x3d.shape[0], 1))], axis = 1).transpose())
                            #x3d = np.dot(R, x3d.transpose())
                            x = x3d.copy()
                            x[2, x[2, :] == 0] = 1
                            x[0] = x[0] / x[2]
                            x[1] = x[1] / x[2]
                            x = x[:2]
                            
                            R2d = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                            x = np.dot(R2d, x).transpose()
                            x[:, 1] = - x[:, 1]
                            x[:, 0] += principal[0]
                            x[:, 1] += principal[1]
                            
                            p3d = np.dot(R, model_pts[int(data['cad_index'][i][0]) - 1].transpose()).transpose()
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
                                print('x', x)
                                print('anchor', anchor[:len(pascalPointInfo[class_name])])
                                print('x3d', x3d, x3d.shape)
                                
                                fig = plt.figure()
                                ax = fig.add_subplot((111),projection='3d')
                                ax.set_xlabel('x')
                                ax.set_ylabel('y')
                                ax.set_zlabel('z')
                                oo = 1
                                xmax, ymax, zmax = oo, oo, oo
                                xmin, ymin, zmin = -oo, -oo, -oo
                                
                                nJoints = len(pascalPointInfo[class_name])
                                
                                points = p3d.transpose()
                                points = points - points.mean(axis = 0)
                                print('points', points)
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
                                
                                img_path = '{}/{}/{}'.format(IMAGE_PATH, obj, im_name)
                                print(img_path)
                                img = cv2.imread(img_path)
                                for j in range(nJoints):
                                    cv2.circle(img, (int(p3d[j, 0]), int(p3d[j, 1])), 6, tuple(np.array((255, 0, 0)) * 1.0 * j / nJoints), -1)
                                for j in range(nJoints):
                                    cv2.circle(img, (int(x[j, 0]), int(x[j, 1])), 4, (0, 0, 255), -1)
                                for j in range(nJoints):
                                    cv2.circle(img, (int(anchor[j, 0]), int(anchor[j, 1])), 2, (0, 255, 255), -1)
                                cv2.imshow('img', img)
                                cv2.waitKey()
                                plt.show()
                        except:
                            pass
                    else:
                        continue
                    
                    if ProjectPoint:
                        nProjectedAnchor += 1
                        total += 1
                        projected_anchor = np.zeros((maxNKeypoints, 2))
                        anchors_3d = np.zeros((maxNKeypoints, 3))
                        for j in range(x.shape[0]):
                          projected_anchor[j] = x[j].copy()
                          anchors_3d[j] = p3d[j].copy()
                    else:
                        total += 1
                        continue
                        
                    cnt += 1
                    
                    for j in range(anchor.shape[0]):
                        if valid[int(data['cad_index'][i][0]) - 1][j] < 0.5:
                            anchor[j] = anchor[j] * 0 - 1
                            projected_anchor[j] = projected_anchor[j] * 0 - 1
                            anchors_3d[j] = anchors_3d[j] * 0 - 1

                    annot['projected_anchors'] += [projected_anchor]
                    annot['anchors_3d'] += [anchors_3d]
                    annot['cad_index'] += [int(data['cad_index'][i][0])]
                    annot['valid'] += [valid[int(data['cad_index'][i][0]) - 1]]
                    annot['anchors'] += [anchor]
                    annot['status'] += [status]
                    annot['vis'] += [vis]
                    annot['dataset'] += [0 if 'pascal' in obj else 1]
                    annot['class_id'] += [classId[class_name]]
                    annot['space_embedding'] += [model_pts[int(data['cad_index'][i][0]) - 1]]
                    annot['avg_embedding'] += [avgModel]
                    
                    class_ = np.zeros(maxLenName, dtype = np.uint8)
                    for v in range(len(class_name)): 
                        class_[v] = ord(class_name[v])
                    annot['class'] += [class_]
                    
                    imgname_ = np.zeros(22, dtype = np.uint8)
                    for v in range(len(im_name)): 
                        imgname_[v] = ord(im_name[v])
                    annot['imgname'] += [imgname_]
                    
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
        print(obj, cnt)

    print('nProjectedAnchor', nProjectedAnchor , total)
    with h5py.File('../../data/Pascal3D/Pascal3D-{}.h5'.format(split),'w') as f:
        f.attrs['name'] = 'Pascal3D-{}'.format(split)
        for k in keys:
            f[k] = np.array(annot[k])
