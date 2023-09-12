from skimage import io
# from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans  #提升聚类速度
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import torch
import cv2
import time

np.set_printoptions(threshold=np.inf)


img_org = io.imread(r"Siltest/ORG.jpg")
img_org = img_org[:,:,:3]
# img_org = img_org[:, :-1]
# img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)  # 彩图化灰度图

img_knn = io.imread(r"Siltest/KNN.jpg")
img_knn = img_knn[:-2, :]
t1=time.time()
gray = cv2.cvtColor(img_knn, cv2.COLOR_BGR2GRAY)   # 灰度图去除3维
ret, knn_thresh = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)  # 按阈值化为二值图

imageWH = io.imread(r"Siltest/WH.jpg")  # 获取分割图像，尺寸与原图一致
imageWH = imageWH[:, :-1]/255
imageWH = cv2.resize(imageWH, [img_org.shape[1],img_org.shape[0]], interpolation=cv2.INTER_CUBIC)
ret, imageWH_in = cv2.threshold(imageWH, 0.5, 1, cv2.THRESH_BINARY)
imageWH_in=imageWH_in.reshape(-1)
imageWH_out = -1*(imageWH_in - 1)
# image = img_org*imageWH

# io.imsave(r'Siltest/colors_use_10.jpg',image)
data=img_org/255.0
data=data.reshape(-1,3)

#运用10个聚类色表示图片
colors_use=30
t3=time.time()
km = MiniBatchKMeans(colors_use,batch_size=2048)
km.fit(data)
new_data = km.cluster_centers_[km.predict(data)]
t4=time.time()
new_data = cv2.cvtColor((new_data * 255).astype(np.uint8).reshape(img_org.shape), cv2.COLOR_BGR2GRAY).reshape(-1)
img_del = new_data * imageWH_out
color_del = list(set(img_del))  # 提取消去部分的颜色
b=color_del.remove(0.)
color_del_num = np.zeros(len(color_del))  # 消去部分颜色数目统计
for i in range(len(img_del)):
    for j in range(len(color_del)):
        if img_del[i] == color_del[j]:
            color_del_num[j] += 1

# 仅保留数目较多像素
out_sum = np.sum(imageWH_out)
color_threshold = (0.02*out_sum).astype(np.uint8)
color_del_num = color_del_num.tolist()
i = 0
while i < len(color_del):
    if color_del_num[i] <= color_threshold:
        color_del.remove(color_del[i])
        color_del_num.remove(color_del_num[i])
    else:
        i += 1

# 消除全图
for i in range(len(new_data)):
    if new_data[i] in color_del:
        new_data[i]=0
    else:
        new_data[i]=255

image_new = new_data.reshape(img_org.shape[0],img_org.shape[1])
# image_new=image_new[:,:-1]*knn_thresh*255
image_new_convert = np.array(np.round(image_new),dtype='uint8')
t2=time.time()
print(t2-t1, 's')  # 整体时间
print(t4-t3, 's')  # 聚类时间
io.imsave(r'Siltest/colors_use_10.jpg',image_new_convert)

