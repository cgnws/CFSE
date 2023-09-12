from sklearn.cluster import MiniBatchKMeans  #提升聚类速度
from imfillKNN import IMFILL
import numpy as np
import cv2
import time

org_img = cv2.imread(r"Siltest/7ORG.jpg", 1)
# knn_img = cv2.imread(r'Siltest/2KNN.jpg', 0)
# BGin = cv2.imread(r"Siltest/BGin.jpg", 1)
# BGin = cv2.resize(BGin, [org_img.shape[1],org_img.shape[0]], interpolation=cv2.INTER_CUBIC)
# BGout = cv2.imread(r"Siltest/BGout.jpg", 1)
# BGout = cv2.resize(BGout, [org_img.shape[1],org_img.shape[0]], interpolation=cv2.INTER_CUBIC)

imfill1 = cv2.imread(r"Siltest/colors_use_10.jpg", 1)
# imfill1 = cv2.imread(r"Siltest/OUT2.jpg", 1)
imfill1 = cv2.resize(imfill1, [org_img.shape[1],org_img.shape[0]], interpolation=cv2.INTER_CUBIC)
# imfill1 = 255-imfill1

cv2.imwrite(r'Siltest/A1.jpg', imfill1/255*org_img)
