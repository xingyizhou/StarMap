from __future__ import absolute_import
from . import paths 
from .utils import pascal3d_meta
from .utils import objectnet3d_meta
nJoints = 16

outputRes = 64
inputRes = 256

eps = 1e-6
    
momentum = 0.0
weightDecay = 0.0
alpha = 0.99
epsilon = 1e-8

scale = 0.25
rotate = 30
hmGauss = 1
shiftY = 10
padScale = 1.5

rootDir = paths.rootDir
dataDir = '{}/data/'.format(paths.rootDir)
expDir = '{}/exp/'.format(paths.rootDir)

pascal3dDir = paths.pascal3dDir
pascalPointInfo = pascal3d_meta.pascalPointInfo
pascalClasses = pascal3d_meta.pascalClasses

pascalClassId = {v: i for i, v in enumerate(pascalClasses)}
pascalClassName = {i: v for i, v in enumerate(pascalClasses)}
pascalDatasetName = ['pascal', 'imagenet']
nPascalKeypoints = [len(pascalPointInfo[v]) for v in pascalClasses]
pascalNamePointId = {v: {u: j for j, u in enumerate(pascalPointInfo[v])} for v in pascalClasses}
pascalIdPointId = {i: {u: j for j, u in enumerate(pascalPointInfo[v])} for i, v in enumerate(pascalClasses)}
pascalMaxKeypoints = 17

ObjectNet3DPath = paths.ObjectNet3DPath
ObjectNet3DPointInfo = objectnet3d_meta.PointInfo
objectnet3DClasses = objectnet3d_meta.objectnet3DClasses
ObjectNet3DClassId = {v: i for i, v in enumerate(objectnet3DClasses)}
ObjectNet3DClassName = {i: v for i, v in enumerate(objectnet3DClasses)}
ObjectNet3DValObj = ['bed', 'bookshelf', 'calculator', 'cellphone', 'computer', 'door', 'filing_cabinet', 'guitar', 'iron', 'knife', 'microwave', 'pen', 'pot', 'rifle', 'shoe', 'slipper', 'stove', 'toilet', 'tub', 'wheelchair']
ObjectNet3DMaxKeypoints = 24



