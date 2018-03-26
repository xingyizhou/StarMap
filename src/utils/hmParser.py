import numpy as np

def nms(det, size = 3):
  pool = np.zeros(det.shape)
  for i in range(size / 2, det.shape[0] - size / 2):
    for j in range(size / 2, det.shape[1] - size / 2):
      pool[i, j] = max(det[i - 1, j - 1], det[i - 1, j], det[i - 1, j + 1], \
                       det[i, j - 1], det[i, j], det[i, j + 1], \
                       det[i + 1, j - 1], det[i + 1, j], det[i + 1, j + 1])
                          
  pool[pool != det] = 0
  return pool

def parseHeatmap(hm, thresh = 0.05):
  assert (len(hm.shape) == 3), hm.shape
  det = hm[0]
  det[det < thresh] = 0
  det = nms(det)
  pts = np.where(det > 0)
  return pts
  

