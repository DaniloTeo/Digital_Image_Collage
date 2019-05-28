import cv2
from cv2 import saliency
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./Corel1000/5/59.jpg")

sal = cv2.saliency.StaticSaliencySpectralResidual_create()
map = sal.computeSaliency(img)
map = (map[1] * 255).astype("uint8")

sal2 = cv2.saliency.StaticSaliencyFineGrained_create()
map2 = sal2.computeSaliency(img)
map2 = (map2[1] * 255).astype("uint8")

thresh = cv2.threshold(map.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh2 = cv2.threshold(map2.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('image antes', thresh2)
contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(contours)
x, y = thresh2.shape
arr = np.zeros((x, y, 3), np.uint8)
final_contours = []
for i in range(len(contours)):
	cnt = contours[i]
	if cv2.contourArea(cnt) > 35000 and cv2.contourArea(cnt) < 15000:
		cv2.drawContours(img, [cnt], -1, [0, 255, 255])
		cv2.fillConvexPoly(arr, cnt, [255, 255, 255])
		final_contours.append(cnt)
cv2.imshow('arr', arr)

# for i in range(img.shape[0]):
# 	for j in range(img.shape[1]):
# 		if thresh3[i][j] == 0:
# 			threshimg[i][j] = [0,0,0]

# cv2.imshow("image", img)
# cv2.imshow("saliency spectral", map)
# cv2.imshow("saliency fine grained", map2)

# cv2.imshow("threshold 1", thresh)
# cv2.imshow("threshold 2", thresh2)

cv2.waitKey(0)