import cv2
from cv2 import saliency
import numpy as np
from numpy.random import randint
# TO DO:
# Realizar cortes em todas as imagens do dataset e
# pasta 10, 9, 7, 3, 

bg = cv2.imread("bg.jpg")
bg = cv2.resize(bg, (int(bg.shape[1] * 0.15),int(bg.shape[0] * 0.15)))

for file in range(100):
	print(file)
	img = cv2.imread("./Corel1000/10/"+str(file)+".jpg")
	auximg = np.array(img)
	sal = cv2.saliency.StaticSaliencySpectralResidual_create()
	map = sal.computeSaliency(img)
	map = (map[1] * 255).astype("uint8")

	thresh = cv2.threshold(map.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	#cv2.imshow('image antes', thresh)

	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	x, y = thresh.shape

	arr = np.zeros((x, y, 3), np.uint8)

	for i in range(len(contours)):
		cnt = contours[i]
		if cv2.contourArea(cnt) > 0.05 * img.shape[0] * img.shape[1]:
			cv2.fillConvexPoly(arr, cnt, [255, 255, 255])
			
	#cv2.imshow('arr', arr)
	threshimg = np.array(img)
	
	aux = np.copy(bg)

	randx = randint(low=0, high=bg.shape[0] - threshimg.shape[0])
	randy = randint(low=0, high=bg.shape[1] - threshimg.shape[1])

	for i in range(randx, randx + img.shape[0]):
		for j in range(randy, randy + img.shape[1]):
			if arr[i - randx][j - randy][0] != 0 and arr[i - randx][j - randy][1] != 0 and arr[i - randx][j - randy][2] != 0:
				aux[i][j] = threshimg[i - randx][j - randy]


	cv2.imwrite('./Imagens_Teste/' + str(file) + '.jpg', aux)



cv2.waitKey(0)