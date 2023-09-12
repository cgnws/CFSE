import numpy as np
import cv2
import os
import time
from silextract_test import SILEXTRACT


def cv_show(name,img):
    # 传入自定义图像名，即图像变量
    cv2.imshow(name,img)
    # 图片不会自动消失
    cv2.waitKey(0)
    # 手动关闭窗口
    cv2.destroyWindow()


if __name__ == "__main__":
    BGin = cv2.imread(r"Siltest/BGin.jpg", 0)
    BGout = cv2.imread(r"Siltest/BGout.jpg", 0)  # 绝对人体区域分割
    _ = SILEXTRACT(BGin, BGout)

    file_root = r"runs\track"
    file1 = r"exp2\crops\person\11"
    file2 = r"exp3\crops\person\11"
    file3 = r"exptest\crops\person\11"
    fileORG = os.path.join(file_root, file1)
    fileKNN = os.path.join(file_root, file2)
    fileSIL = os.path.join(file_root, file3)
    t1 = time.time()

    for root, dirs, files in os.walk(fileORG):
        # root 表示当前访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        if not os.path.exists(fileSIL):
            os.makedirs(fileSIL)
        # 遍历文件
        for f in files:
            org_video = os.path.join(fileORG, f)
            KNN_video = os.path.join(fileKNN, f)
            SIL_video = os.path.join(fileSIL, f)
            org_img = cv2.imread(org_video, 1)
            knn_img = cv2.imread(KNN_video, 0)
            fin_img=_.silextract(org_img, knn_img)

            cv2.imwrite(SIL_video, fin_img)

    t2 = time.time()
    print(t2-t1)
