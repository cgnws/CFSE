import cv2
import os
import numpy as np
import time

class IMFILL(object):
    def __init__(self):
        self.ratio = 0.02

    def EDGE_DETECTION_Canny(self, grayImage):
        Canny=cv2.Canny(grayImage,64,128)
        return Canny

    def ContourKNNFG(self, img_gray, img_seg):  # 利用轮廓获取部分前景
        EDimg = self.EDGE_DETECTION_Canny(img_gray)
        conv_EDimg = 255-EDimg
        conv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.erode(conv_EDimg, conv_kernel)  # 腐蚀

        contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv_contours = []
        for contour in contours:
            if contour.shape[0] > 4:
                _contour = contour.reshape(-1, 2)
                contour_x = _contour[:, 0]
                contour_y = _contour[:, 1]
                j = 0  # 计算轮廓线与SEG图像重合的点
                edge_num = np.sum(contour == 0) + np.sum(contour == 127)  # 计算图像边界轮廓点数目
                for i in range(contour.shape[0]):
                    if img_seg[contour_y[i],contour_x[i]] > 0:
                        j+=1

                if j/contour.shape[0] > 0.6:  # 轮廓线重合点的阈值
                    if edge_num <= 3:
                        cv_contours.append(contour)

        new_img = np.zeros(img.shape)
        cv2.fillPoly(new_img, cv_contours, (1,1,1))
        new_img = cv2.dilate(new_img, conv_kernel) # 膨胀

        return new_img

    def ContourGRAYFG(self, img_gray, img_fg):  # 利用轮廓获取部分前景
        EDimg = self.EDGE_DETECTION_Canny(img_gray)
        conv_EDimg = 255-EDimg
        conv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.erode(conv_EDimg, conv_kernel)  # 腐蚀

        contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv_contours = []
        for contour in contours:
            if contour.shape[0] > 3:
                _contour = contour.reshape(-1, 2)
                contour_x = _contour[:, 0]
                contour_y = _contour[:, 1]
                j = 0 # 计算轮廓线与SEG图像重合的点
                edge_num = np.sum(contour == 0) + np.sum(contour == 127)  # 计算图像边界轮廓点数目
                for i in range(contour.shape[0]):
                    if img_fg[contour_y[i],contour_x[i]] > 0:
                        j+=1

                if j/contour.shape[0] > 0.5:  # 轮廓线重合点的阈值
                    if edge_num <= 3:
                        cv_contours.append(contour)

        new_img = np.zeros(img.shape)
        cv2.fillPoly(new_img, cv_contours, (1,1,1))
        new_img = cv2.dilate(new_img, conv_kernel)  # 膨胀

        return new_img

    def BGjoin(self, img_gray, img_knn, img_seg):
        ContourFG_knn = self.ContourKNNFG(img_knn, img_seg)
        ContourFG_gray = self.ContourGRAYFG(img_gray, ContourFG_knn*255)
        # ContourBG_knn = self.ContourBG(img_knn)
        # ContourBG_gray = self.ContourBG(img_gray)
        # ContourFG_join = ContourFG_knn + ContourFG_gray
        ContourFG_join = np.logical_or(ContourFG_knn, ContourFG_gray)
        # ContourFG_join[ContourFG_join>0] = 1

        # cv2.imwrite(r'Siltest/IN15.jpg', ContourFG_knn * 255)
        # cv2.imwrite(r'Siltest/IN16.jpg', ContourFG_gray * 255)
        # cv2.imwrite(r'Siltest/IN17.jpg', ContourFG_join * 255)

        return 1 - ContourFG_join


if __name__ == "__main__":

    img_gray = cv2.imread(r'Siltest/6ORG.jpg', 0)
    img_knn = cv2.imread(r'Siltest/6KNN.jpg', 0)
    img_seg = cv2.imread(r'Siltest/6SEG.jpg', 0)
    # _, img_knn = cv2.threshold(img_knn, 127, 255, cv2.THRESH_BINARY)
    BGin = cv2.imread(r"Siltest/BGin.jpg", 0)  # 绝对人体区域分割
    BGin = cv2.resize(BGin, [img_knn.shape[1],img_knn.shape[0]], interpolation=cv2.INTER_CUBIC)
    _, BGin = cv2.threshold(BGin, 127, 1, cv2.THRESH_BINARY)
    t1 = time.time()
    a = IMFILL(BGin)
    ContourBG = a.BGjoin(img_gray, img_knn, img_seg)
    t2 = time.time()

    print(t2-t1)

    cv2.imwrite(r'Siltest/imfill.jpg', ContourBG*255)
