import cv2
import os
import numpy as np

def EDGE_DETECTION_Canny(grayImage):
    Canny=cv2.Canny(grayImage,64,128)
    return Canny

BGin = cv2.imread(r'Siltest/BGin.jpg', 0)
_, BGin = cv2.threshold(BGin, 127, 255, cv2.THRESH_BINARY)
imfill = cv2.imread(r'Siltest/colors_use_10.jpg', 0)
EDimg = 255 - EDGE_DETECTION_Canny(imfill)
# _, imfill = cv2.threshold(imfill, 127, 255, cv2.THRESH_BINARY)
conv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
EDimg = cv2.morphologyEx(EDimg,cv2.MORPH_OPEN,conv_kernel)  # 开运算，先腐蚀再膨胀，先扩黑
# EDimg = cv2.morphologyEx(EDimg,cv2.MORPH_CLOSE,conv_kernel)  # 闭运算，先膨胀再腐蚀，先扩白
# imfill = cv2.erode(imfill, conv_kernel)  # 腐蚀
# # imfill = cv2.dilate(conv_kernel, imfill)  # 膨胀
# cv2.imwrite(r'Siltest/imfill1.jpg', imfill)


cv2.imwrite(r'Siltest/imfill1.jpg', EDimg)

contours, _ = cv2.findContours(EDimg.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
num_cpoint = [x.shape[0] for x in contours]
cmax = max(num_cpoint)
id_cmax = num_cpoint.index(cmax)
cv_contour = contours[id_cmax]
cv_contours = []
for contour in contours:
    _contour = contour.reshape(-1, 2)
    contour_x = _contour[:,0]
    contour_y = _contour[:,1]
    area = cv2.contourArea(contour)
    if max(contour_x)>=imfill.shape[1]-1 or min(contour_x)==0 or max(contour_y)>=imfill.shape[0]-1 or min(contour_y)==0:
        inc_num = 0
        # for i in range(len(contour)):
        #     if BGin[contour[i][0][1], contour[i][0][0]] == 255:
        #         inc_num += 1
        #         break

        if inc_num == 0:
            cv_contours.append(contour)

new_img = np.zeros(imfill.shape)
cv2.fillPoly(new_img, [cv_contour], (255, 255, 255))
# new_img = cv2.dilate(new_img, conv_kernel)  # 膨胀
gray_img = cv2.imread(r"Siltest/14ORG.jpg", 0)
cv2.imwrite(r'Siltest/imfill1.jpg', new_img/255*gray_img)