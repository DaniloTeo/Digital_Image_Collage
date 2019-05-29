import cv2
from cv2 import saliency
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./Corel1000/9/79.jpg")
auximg = np.array(img)
sal = cv2.saliency.StaticSaliencySpectralResidual_create()
map = sal.computeSaliency(img)
map = (map[1] * 255).astype("uint8")

thresh = cv2.threshold(map.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cv2.imshow('image antes', thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

x, y = thresh.shape

arr = np.zeros((x, y, 3), np.uint8)

for i in range(len(contours)):
	cnt = contours[i]
	if cv2.contourArea(cnt) > 0.05 * img.shape[0] * img.shape[1]:
		cv2.fillConvexPoly(arr, cnt, [255, 255, 255])
		
cv2.imshow('arr', arr)
threshimg = np.array(img)

for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		if arr[i][j][0] == 0 and arr[i][j][1] == 0 and arr[i][j][2] == 0:
			threshimg[i][j] = [0,0,0]

cv2.imshow('cut', threshimg)

# cv2.imshow("image", img)
# cv2.imshow("saliency spectral", map)
# cv2.imshow("saliency fine grained", map2)

# cv2.imshow("threshold 1", thresh)
# cv2.imshow("threshold 2", thresh2)

cv2.waitKey(0)