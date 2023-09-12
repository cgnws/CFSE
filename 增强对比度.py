import cv2
import numpy as np
import time

def EDGE_DETECTION_Roberts(grayImage):
    # 自定义Roberts算子
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    #按照相同的权重相加
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    Roberts[Roberts<20] = 0
    Roberts[Roberts>0] = 255
    return Roberts

def EDGE_DETECTION_Prewitt(grayImage):
    # 自定义Prewitt算子
    kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    #按照相同的权重相加
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    Prewitt[Prewitt<60] = 0
    Prewitt[Prewitt>0] = 255
    return Prewitt

def EDGE_DETECTION_Sobel(grayImage):
    x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)
    # 转 uint8 ,图像融合
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    conv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    Sobel = cv2.morphologyEx(Sobel,cv2.MORPH_OPEN,conv_kernel)  # 开运算，先腐蚀再膨胀，先扩黑

    Sobel[Sobel<40] = 0
    Sobel[Sobel>0] = 255
    return Sobel

def EDGE_DETECTION_Canny(grayImage):
    Canny=cv2.Canny(grayImage,16,128)
    return Canny

def EDGE_DETECTION_Laplacian(grayImage):
    Laplacian=cv2.Laplacian(grayImage,cv2.CV_16S)
    Laplacian[Laplacian<15] = 0
    Laplacian[Laplacian>0] = 255
    return Laplacian


# 用来正常显示中文标签
t1 = time.time()
grayImage = cv2.imread(r"Siltest/ORG.jpg", 0)
# 边缘检测 Roberts Prewitt Sobel Canny Laplacian
# 主要用后三种
EDimg = EDGE_DETECTION_Canny(grayImage)
cv2.imwrite(r'Siltest/Canny.jpg', EDimg)

# org_img = cv2.imread(r"Siltest/colors_use_10.jpg", 0)
# conv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9, 9))
# # org_img = cv2.morphologyEx(org_img,cv2.MORPH_OPEN,conv_kernel)  # 开运算，先腐蚀再膨胀，先扩黑
# img = cv2.morphologyEx(org_img,cv2.MORPH_CLOSE,conv_kernel)  # 闭运算，先膨胀再腐蚀，先扩白
#
# org_img[org_img<127] = 0
# org_img[org_img>0] = 255
# #绘制彩色图轮廓线帮忙分割图像
# # gray_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
# NEWimg = np.zeros(org_img.shape)
# contours, _ = cv2.findContours(org_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cv_contours = []
# for contour in contours:
#     if len(contour)>10:
#         new_img = np.zeros(img.shape)
#         cv2.fillPoly(new_img, contour, (255, 255, 255))
#         area = cv2.contourArea(contour)
#         if area > 70:
#             cv_contours.append(contour)
#         # cv2.imwrite(r'Siltest/colors_use_101.jpg', new_img)
#         a=1
#
#     area = cv2.contourArea(contour)
#
#     # if area >= area_thresh:
#     #     cv_contours.append(contour)
# # cv2.drawContours(NEWimg,cv_contours,-1,(255,255,255),1)
# cv2.fillPoly(NEWimg, cv_contours, (255, 255, 255))
# cv2.imwrite(r'Siltest/colors_use_101.jpg', NEWimg)