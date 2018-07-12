import torch.utils.data as data
import numpy as np
import ref
import torch
from h5py import File
import cv2
from utils.utils import Rnd, Flip
from utils.img import Crop, DrawGaussian, Transform, Transform3D

class ObjectNet3D(data.Dataset):
  def __init__(self, opt, split):
    print('==> initializing ObjectNet3D Star {} data.'.format(split))
    annot = {}
    tags = ['bbox', 'anchors', 'vis', 'dataset', 'class_id', 'imgname', 
            'viewpoint_azimuth', 'viewpoint_elevation', 'viewpoint_theta', 'anchors_3d', 
            'space_embedding', 'truncated', 'occluded', 'difficult', 'valid']
    f = File('{}/ObjectNet3D/ObjectNet3D-{}.h5'.format(ref.dataDir, split), 'r')
    for tag in tags:
      annot[tag] = np.asarray(f[tag]).copy()
    f.close()
    annot['index'] = np.arange(len(annot['class_id']))
    tags = tags + ['index']

    inds = []
    if split == 'train':
      for i in range(len(annot['class_id'])):
        if opt.ObjectNet3DTrainAll or not (ref.ObjectNet3DClassName[annot['class_id'][i]] in ref.ObjectNet3DValObj):
          inds.append(i)
      self.nSup = len(inds)
    else:
      inds_sup, inds_novel = [], []
      for i in range(len(annot['class_id'])):
        if annot['truncated'][i] < 0.5 and annot['occluded'][i] < 0.5 and annot['difficult'][i] < 0.5:
          if not (ref.ObjectNet3DClassName[annot['class_id'][i]] in ref.ObjectNet3DValObj):
            inds_sup.append(i)
          else:
            inds_novel.append(i)
      self.nSup = len(inds_sup)
      inds = inds_sup + inds_novel
    
    for tag in tags:
      annot[tag] = annot[tag][inds]

    self.split = split
    self.opt = opt
    self.annot = annot
    self.nSamples = len(annot['vis'])
    print('Loaded ObjectNet3D {} {} samples'.format(split, self.nSamples))
  
  def LoadImage(self, index):
    img_name = ''
    for v in range(len(self.annot['imgname'][index])):
      c = self.annot['imgname'][index][v]
      if c != 0:
        img_name += chr(c)
    path = '{}/Images/{}.JPEG'.format(ref.ObjectNet3DPath, img_name)
    img = cv2.imread(path)
    return img
  
  
  def GetPartInfo(self, index):
    pts2d = self.annot['anchors'][index].copy()
    pts3d = self.annot['anchors_3d'][index].copy()
    emb = self.annot['space_embedding'][index].copy()
    box = self.annot['bbox'][index].copy()
    c = np.array([(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.])
    s = max((box[2] - box[0]), (box[3] - box[1])) * ref.padScale
    return pts2d, pts3d, emb, c, s
      
  def __getitem__(self, index):
    img = self.LoadImage(index)
    pts2d, pts3d, emb, c, s = self.GetPartInfo(index)
    s = min(s, max(img.shape[0], img.shape[1])) * 1.0
    pts3d[:, 2] += s / 2
    
    r = 0
    if self.split == 'train':
      s = s * (2 ** Rnd(ref.scale))
      c[1] = c[1] + Rnd(ref.shiftY)
      r = 0 if np.random.random() < 0.6 else Rnd(ref.rotate)
    inp = Crop(img, c, s, r, ref.inputRes)
    inp = inp.transpose(2, 0, 1) / 256.
    starMap = np.zeros((1, ref.outputRes, ref.outputRes))
    embMap = np.zeros((3, ref.outputRes, ref.outputRes))
    depMap = np.zeros((1, ref.outputRes, ref.outputRes))
    mask = np.concatenate([np.ones((1, ref.outputRes, ref.outputRes)), np.zeros((4, ref.outputRes, ref.outputRes))]);

    for i in range(pts3d.shape[0]):
      if self.annot['valid'][index][i] > ref.eps:
        if self.annot['vis'][index][i] > ref.eps:
          pt2d = Transform(pts2d[i], c, s, r, ref.outputRes).astype(np.int32)
          pt3d = Transform3D(pts3d[i], c, s, r, ref.outputRes).astype(np.int32)
          if pt2d[0] >= 0 and pt2d[0] < ref.outputRes and pt2d[1] >=0 and pt2d[1] < ref.outputRes:
            embMap[:, pt2d[1], pt2d[0]] = emb[i]
            depMap[0, pt2d[1], pt2d[0]] = min(max(1.0 * pt3d[2] / ref.outputRes - 0.5, -2), 2)
            mask[1:, pt2d[1], pt2d[0]] = 1
          starMap[0] = np.maximum(starMap[0], DrawGaussian(np.zeros((ref.outputRes, ref.outputRes)), pt2d, ref.hmGauss).copy())
    
    out = starMap
    if 'emb' in self.opt.task:
      out = np.concatenate([out, embMap])
    if 'dep' in self.opt.task:
      out = np.concatenate([out, depMap])
    mask = mask[:out.shape[0]].copy()
    
    if self.split == 'train':
      if np.random.random() < 0.5:
        inp = Flip(inp)
        out = Flip(out)
        mask = Flip(mask)
        if 'emb' in self.opt.task:
          out[1] = - out[1]
    return inp, out, mask
    
  def __len__(self):
    return self.nSamples

