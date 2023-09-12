import os
import cv2
import numpy as np


if __name__ == '__main__':

    # img = cv2.imread("peopleresize/WH.jpg")
    img = np.zeros([116,54])
    pts = np.array([[25,2],
                    [16,8],
                    [3,39],
                    [5,69],
                    [9,70],
                    [9,83],
                    [5,97],
                    [3,99],
                    [3,109],
                    [17,113],
                    [34,113],
                    [50,106],
                    [50,102],
                    [45,72],
                    [51,65],
                    [48,26],
                    [40,7],
                    [32,2]])

    # pts = np.array([[25,20],
    #                 [25,23],
    #                 [18,28],
    #                 [13,38],
    #                 [17,42],
    #                 [13,52],
    #                 [21,90],
    #                 [27,79],
    #                 [30,75],
    #                 [33,87],
    #                 [35,69],
    #                 [38,59],
    #                 [43,52],
    #                 [44,34],
    #                 [35,27],
    #                 [35,21]])

    # cv2.fillPoly(img,[pts],[255,255,255])#填充多边形
    cv2.fillPoly(img,[pts],255)#填充多边形
    '''生成图片存储的目标路径'''
    save_path = 'peopleresize/'
    save_img = save_path + "BGout.jpg"
    cv2.imwrite(save_img, img)
