import numpy as np
import cv2
# from sklearn.mixture import GaussianMixture
import time

cap = cv2.VideoCapture("9_9test/d2s_5m.mp4")
# cap = cv2.VideoCapture(0)
# fgbg = cv2.createBackgroundSubtractorMOG2()
# fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
fgbg = cv2.createBackgroundSubtractorKNN()

# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()  # 不合适
# fgbg = cv2.bgsegm.createBackgroundSubtractorLSBP()  # 不适合
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # GMG
# fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()  # GMG 不合适

# GMM = GaussianMixture(n_components=3)
while(1):
    t1=time.time()
    ret, frame = cap.read()
    t2 = time.time()
    fgmask = fgbg.apply(frame)
    t3 = time.time()
    print("video:",t2-t1)
    print("sub:",t3 - t2)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)  # GMG
    cv2.imshow('frame', fgmask)
    # cv2.imshow('original', frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()