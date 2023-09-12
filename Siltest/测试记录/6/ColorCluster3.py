#!D:\Anaconda3\envs\py388test\ python
# coding=utf-8
from skimage import io
from sklearn.cluster import MiniBatchKMeans  #提升聚类速度
import numpy as np
import torch
import cv2
import time

# 数据准备----------------------------------------------------------------------
img_org = io.imread(r"Siltest/ORG.jpg")
img_org = img_org[:, :, :3]
org_shape0 = img_org.shape[0]
org_shape1 = img_org.shape[1]

imageWH = io.imread(r"Siltest/WH.jpg")  # 绝对背景区域分割
imageWH = cv2.resize(imageWH, [org_shape1, org_shape0], interpolation=cv2.INTER_CUBIC)
ret, imageWH_in = cv2.threshold(imageWH, 127, 1, cv2.THRESH_BINARY)
imageWH_in = imageWH_in.reshape(-1)
imageWH_out = 1-imageWH_in

imageWH2 = io.imread(r"Siltest/WH2.jpg")  # 绝对人体区域分割
imageWH2 = cv2.resize(imageWH2, [org_shape1, org_shape0], interpolation=cv2.INTER_CUBIC)
ret2, imageWH2_in = cv2.threshold(imageWH2, 127, 1, cv2.THRESH_BINARY)
imageWH2_in = imageWH2_in.reshape(-1)

data = img_org.reshape(-1, 3)

t1=time.time()
#K-mean聚类压缩颜色--------------------------------------------------------------------------
#运用10个聚类色表示图片
colors_use = 30
t3=time.time()
km = MiniBatchKMeans(colors_use,batch_size=2048)
km.fit(data)
new_data = km.cluster_centers_[km.predict(data)]
t4 = time.time()
# 压缩颜色后转化为灰度图减少运算
new_data = cv2.cvtColor(new_data.astype(np.uint8).reshape(img_org.shape), cv2.COLOR_BGR2GRAY).reshape(-1)

#黑白图外侧颜色采集，与颜色统计-----------------------------------------------------
img_del = new_data * imageWH_out
color_del = list(set(img_del))  # 提取消去部分的颜色
color_del_num = [0 for i in color_del]  # 消去部分颜色数目统计
for i in range(len(img_del)):
    if img_del[i] != 0:
        for j in range(len(color_del)):
            if img_del[i] == color_del[j]:
                color_del_num[j] += 1

img_hold = new_data * imageWH2_in
color_hold = list(set(img_del))  # 提取消去部分的颜色
color_hold_num = [0 for i in color_hold]  # 消去部分颜色数目统计
for i in range(len(img_hold)):
    if img_hold[i] != 0:
        for j in range(len(color_hold)):
            if img_hold[i] == color_hold[j]:
                color_hold_num[j] += 1

# 仅保留数目较多像素
mask = imageWH_out != 0
out_sum = np.sum(imageWH_out[mask])
color_threshold = (0.02*out_sum).astype(np.uint8)
i = 0
while i < len(color_del):
    if color_del_num[i] <= color_threshold:
        color_del.remove(color_del[i])
        color_del_num.remove(color_del_num[i])
    else:
        i += 1

mask = imageWH2_in != 0
in_sum = np.sum(imageWH2_in[mask])
color_threshold = (0.02*in_sum).astype(np.uint8)
i = 0
while i < len(color_hold):
    if color_hold_num[i] <= color_threshold:
        color_hold.remove(color_hold[i])
        color_hold_num.remove(color_hold_num[i])
    else:
        i += 1

i = 0
while i < len(color_del):
    if color_del[i] in color_hold:
        for j in range(len(color_hold)):
            if color_del[i] == color_hold[j]:
                if color_del_num[i] >= color_hold_num[j]:
                    color_del.remove(color_del[i])
                    color_del_num.remove(color_del_num[i])
                    i -= 1
    i += 1

#颜色消除-------------------------------------------------------------------
# 消除全图
fin_data = imageWH_in
for i in range(len(new_data)):
    if fin_data[i] == 1:
        if new_data[i] in color_del:
            fin_data[i]=0
        else:
            fin_data[i]=255

image_new = fin_data.reshape(org_shape0,org_shape1)

# #开闭运算除噪--------------------------------------------------------------------------------------------
# k = np.ones((2, 2), np.uint8)
# image_new_convert = cv2.morphologyEx(image_new_convert, cv2.MORPH_CLOSE, k)  # 闭运算，先膨胀再腐蚀，先扩白
# # image_new_convert = cv2.morphologyEx(image_new_convert,cv2.MORPH_OPEN,k)  # 开运算，先腐蚀再膨胀，先扩黑


t2=time.time()
print(t2-t1, 's')  # 整体时间
print(t4-t3, 's')  # 聚类时间
io.imsave(r'Siltest/colors_use_10.jpg',image_new)

