import cv2
from cv2 import saliency
import numpy as np
from numpy.random import randint

# k-means para fazer colagem com objetos similares ou diferentes [dataset de stickers]

FOLDER_RANGE = [12, 18, 17, 4, 11, 30, 2, 6, 10, 0]
FOLDER_RANGE_BG = [388, 30, 357, 24, 390, 29]

folder = randint(1,7);print(folder)
file = randint(1, FOLDER_RANGE_BG[folder-1]);print(file)

bg = cv2.imread("./My_Flickr/images/" + str(folder) + "/" + str(file) + ".jpg")
bg = cv2.resize(bg, (int(bg.shape[1] * 0.15),int(bg.shape[0] * 0.15)))
print(f"shape de bg: {bg.shape}")
for n in range(25):
	folder = randint(1,10)
	file = randint(0, FOLDER_RANGE[folder-1])
	img = cv2.imread("./Imagens_Teste/" + str(folder) + "/"+str(file)+".jpg")

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

	if bg.shape[0] < threshimg.shape[0]:
		aux_high_x = threshimg.shape[0]
		aux_low_x = bg.shape[0]
	else:
		aux_high_x = bg.shape[0]
		aux_low_x= threshimg.shape[0]

	if bg.shape[1] < threshimg.shape[1]:
		aux_high_y = threshimg.shape[1]
		aux_low_y = bg.shape[1]

	else:
		aux_high_y = bg.shape[1]
		aux_low_y = threshimg.shape[1]
	
	if n % 4 == 0:
		randx = randint(low=0, high=(aux_high_x - aux_low_x)//2)
		randy = randint(low=0, high=(aux_high_y - aux_low_y)//2)
	elif n % 4 == 1:
		randx = randint(low=(aux_high_x - aux_low_x)//2, high=aux_high_x - aux_low_x)
		randy = randint(low=0, high=(aux_high_y - aux_low_y)//2)
	elif n % 4 == 2:
		randx = randint(low=(aux_high_x - aux_low_x)//2, high=aux_high_x - aux_low_x)
		randy = randint(low=(aux_high_y - aux_low_y)//2, high=aux_high_y - aux_low_y)
	elif n % 4 == 3:
		randx = randint(low=0, high=(aux_high_x - aux_low_x)//2)
		randy = randint(low=(aux_high_y - aux_low_y)//2, high=aux_high_y - aux_low_y)

	for i in range(randx, randx + img.shape[0]):
		for j in range(randy, randy + img.shape[1]):
			if arr[i - randx][j - randy][0] != 0 and arr[i - randx][j - randy][1] != 0 and arr[i - randx][j - randy][2] != 0:
				bg[i][j] = (3 * (threshimg[i - randx][j - randy].astype(float)) + (bg[i][j].astype(float))) // 4

	bg = bg.astype(np.uint8)
	#cv2.imwrite('./Imagens_Teste/' + str(file) + '.jpg', aux)


cv2.imshow('collage', bg)
cv2.waitKey(0)