import cv2
import numpy as np
for i in range(5053, 5062):
  try:
    img = cv2.imread('IMG_{}.jpg'.format(i))
    print(i)
    img = cv2.resize(img, (img.shape[1] / 8, img.shape[0] / 8))
    cv2.imwrite('IMG_{}.jpg'.format(i), img)
  except:
    pass