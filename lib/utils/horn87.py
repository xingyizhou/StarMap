import numpy as np

def horn87(pointsS, pointsT, weight = None):
  centerS = pointsS.mean(axis = 1)
  centerT = pointsT.mean(axis = 1)
  for i in range(pointsS.shape[1]):
    pointsS[:, i] = pointsS[:, i] - centerS
    pointsT[:, i] = pointsT[:, i] - centerT
  if not (weight is None):
    for i in range(pointsS.shape[1]):
      pointsS[:, i] *= weight[i] ** 0.5
      pointsT[:, i] *= weight[i] ** 0.5
  
  M = np.dot(pointsS, pointsT.transpose(1, 0))
  N = np.array([[M[0, 0] + M[1, 1] + M[2, 2], M[1, 2] - M[2, 1], M[2, 0] - M[0, 2], M[0, 1] - M[1, 0]], 
                [M[1, 2] - M[2, 1], M[0, 0] - M[1, 1] - M[2, 2], M[0, 1] + M[1, 0], M[0, 2] + M[2, 0]], 
                [M[2, 0] - M[0, 2], M[0, 1] + M[1, 0], M[1, 1] - M[0, 0] - M[2, 2], M[1, 2] + M[2, 1]], 
                [M[0, 1] - M[1, 0], M[2, 0] + M[0, 2], M[1, 2] + M[2, 1], M[2, 2] - M[0, 0] - M[1, 1]]])
  v, u = np.linalg.eig(N)
  id = v.argmax()

  q = u[:, id]
  R = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])], 
                [2*(q[2]*q[1]+q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*(q[2]*q[3]-q[0]*q[1])], 
                [2*(q[3]*q[1]-q[0]*q[2]), 2*(q[3]*q[2]+q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]])

  s = (pointsT * np.dot(R, pointsS)).sum() / (pointsS * pointsS).sum()
  t = centerT - s * np.dot(R, centerS)
  return R.astype(np.float32), t.astype(np.float32), s

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
