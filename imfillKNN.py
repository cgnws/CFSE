import cv2
import os
import numpy as np
import time

class IMFILL(object):
    def __init__(self, BGin):
        self.BGin = BGin
        self.newBGin = None
        self.ratio = 0.02

    def imfill1(self, img):  # img是输入图像，ratio是须保留的轮廓面积比率,要把黑色分隔开
        conv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        # img = cv2.morphologyEx(img,cv2.MORPH_OPEN,conv_kernel)  # 开运算，先腐蚀再膨胀，先扩黑
        img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,conv_kernel)  # 闭运算，先膨胀再腐蚀，先扩白
        # img = cv2.erode(img, conv_kernel)  # 腐蚀
        # img = cv2.dilate(img, conv_kernel)  # 膨胀
        #blur image to reduce the noise in the image while thresholding 均值滤波
        img = cv2.blur(img, (3,3))
        #Apply thresholding to the image
        # ret, img = cv2.threshold(img, 0.1, 1, cv2.THRESH_BINARY)
        img[img<80]=0
        img[img>0]=1

        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        n = len(contours)  # 轮廓的个数
        cv_contours = []
        area_thresh = self.ratio * img.shape[0]*img.shape[1]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= area_thresh:
                cv_contours.append(contour)
            else:
                continue
        new_img = np.zeros(img.shape)
        cv2.fillPoly(new_img, cv_contours, (1, 1, 1))

        return new_img

    def imfill2(self, img, newBGin):  # img是输入图像，ratio是须保留的轮廓面积比率
        #Apply thresholding to the image
        # ret, img = cv2.threshold(img, 254, 1, cv2.THRESH_OTSU)

        conv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        img = cv2.morphologyEx(img,cv2.MORPH_OPEN,conv_kernel)  # 开运算，先腐蚀再膨胀，先扩黑
        # img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,conv_kernel)  # 闭运算，先膨胀再腐蚀，先扩白
        img = cv2.erode(img, conv_kernel)  # 腐蚀
        # img = cv2.dilate(img, conv_kernel)  # 膨胀

        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv_contours = []
        for contour in contours:
            inc_num = 0
            for i in range(len(contour)):
                if newBGin[contour[i][0][1], contour[i][0][0]] == 1:
                    inc_num += 1
            area_ratio = inc_num/len(contour)

            if area_ratio > 0.1:
                cv_contours.append(contour)

        new_img = np.zeros(img.shape)
        cv2.fillPoly(new_img, cv_contours, (1, 1, 1))
        new_img = cv2.dilate(new_img, conv_kernel)  # 膨胀


        return new_img

    def imfill(self, img):
        img1 = self.imfill1(img)
        # cv2.imwrite(r'Siltest/imfill.jpg', img1*255)

        hole_rate = 0
        if hole_rate < 0.9:
            mask = 1-img1
            img2 = self.imfill2(mask.astype(np.uint8), self.BGin)
            fin_img = img1+img2
            fin_img = cv2.blur(fin_img, (3,3))
            _, fin_img = cv2.threshold(fin_img, 0.5, 1, cv2.THRESH_BINARY)
        else:
            fin_img = img1
        conv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        # fin_img = cv2.morphologyEx(fin_img,cv2.MORPH_OPEN,conv_kernel)  # 开运算，先腐蚀再膨胀，先扩黑
        fin_img = cv2.morphologyEx(fin_img,cv2.MORPH_CLOSE,conv_kernel)  # 闭运算，先膨胀再腐蚀，先扩白


        return fin_img


if __name__ == "__main__":

    img = cv2.imread(r'Siltest/14KNN.jpg', 0)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    BGin = cv2.imread(r"Siltest/BGin.jpg", 0)  # 绝对人体区域分割
    BGin = cv2.resize(BGin, [img.shape[1],img.shape[0]], interpolation=cv2.INTER_CUBIC)
    _, BGin = cv2.threshold(BGin, 127, 1, cv2.THRESH_BINARY)
    t1 = time.time()
    a = IMFILL(BGin)
    fin_img=a.imfill(img)
    t2 = time.time()
    print(t2-t1)

    cv2.imwrite(r'Siltest/imfill1.jpg', fin_img*255)
