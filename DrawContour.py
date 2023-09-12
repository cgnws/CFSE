import cv2 as cv
#read the image
img = cv.imread(r'Siltest/colors_use_10.jpg')
#convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#blur image to reduce the noise in the image while thresholding
blur = cv.blur(gray, (3,3))
#Apply thresholding to the image
ret, thresh = cv.threshold(blur, 127, 255, cv.THRESH_OTSU)
#find the contours in the image
contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv_contours = []
for contour in contours:
    area = cv.contourArea(contour)
    if area >= 100:
        cv_contours.append(contour)
        # x, y, w, h = cv2.boundingRect(contour)
        # img[y:y + h, x:x + w] = 255
    else:
        continue

#draw the obtained contour lines(or the set of coordinates forming a line) on the original image
cv.drawContours(img, cv_contours, -1, (0,255,0), 1)
#show the image
cv.namedWindow('Contours',cv.WINDOW_NORMAL)
cv.namedWindow('Thresh',cv.WINDOW_NORMAL)
cv.imshow('Contours', img)
cv.imshow('Thresh', thresh)
if cv.waitKey(0):
    cv.destroyAllWindows()