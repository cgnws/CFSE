import numpy as np
import cv2
import os
import time
from silextract_test import SILEXTRACT
from multiprocessing import Pool


def run_BG(sile, fileORG, fileKNN, fileSEG, fileSIL, dir):
    fileORG_p = os.path.join(fileORG, dir)
    fileKNN_p = os.path.join(fileKNN, dir)
    fileSEG_p = os.path.join(fileSEG, dir)
    fileSIL_p = os.path.join(fileSIL, dir)
    if not os.path.exists(fileSIL_p):
        os.makedirs(fileSIL_p)
    for root, dirs, files in os.walk(fileORG_p):
        for file in files:
            org_video = os.path.join(fileORG_p, file)
            KNN_video = os.path.join(fileKNN_p, file)
            SEG_video = os.path.join(fileSEG_p, file)
            SIL_video = os.path.join(fileSIL_p, file)
            org_img = cv2.imread(org_video, 1)
            knn_img = cv2.imread(KNN_video, 0)
            seg_img = cv2.imread(SEG_video, 0)
            fin_img=sile.silextract(org_img, knn_img, seg_img)

            cv2.imwrite(SIL_video, fin_img)



if __name__ == "__main__":
    pool = Pool(2)  # cpu 4个逻辑内核，进程数设置为4
    BGin = cv2.imread(r"Siltest/BGin.jpg", 0)
    BGout = cv2.imread(r"Siltest/BGout.jpg", 0)  # 绝对人体区域分割
    sile = SILEXTRACT(BGin, BGout)

    file_root = r"runs\track"
    file1 = r"exp2\crops\person"  # ORG
    file2 = r"exp3\crops\person"  # KNN
    file3 = r"exp4\crops\person"  # seg
    file4 = r"exptest2\crops\person" # 输出文件夹
    fileORG = os.path.join(file_root, file1)
    fileKNN = os.path.join(file_root, file2)
    fileSEG = os.path.join(file_root, file3)
    fileSIL = os.path.join(file_root, file4)
    t1 = time.time()

    for root, dirs, files in os.walk(fileORG):
        # root 表示当前访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        for dir in dirs:
            # 遍历文件
            pool.apply_async(func=run_BG, args=(sile, fileORG, fileKNN, fileSEG, fileSIL, dir))

    # 注意：join必须放在close()后面，否则将不会等待子进程打印结束，而直接结束
    pool.close()  # 关闭进程池
    pool.join()  # 进程池中进程执行完毕后再关闭，如果注释，那么程序直接关闭

    t2 = time.time()
    print(t2-t1)
