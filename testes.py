import cv2
from cv2 import saliency
import numpy as np
from numpy.random import randint

# k-means para fazer colagem com objetos similares ou diferentes [dataset de stickers]

# Folder sizes for the Imagens_Teste/ folder
FOLDER_RANGE = [12, 18, 17, 4, 11, 30, 2, 6, 10, 0]

# Folder sizes for the My_Flickr/ folder
FOLDER_RANGE_BG = [388, 30, 357, 24, 390, 28]

folder = randint(1,7);
file = randint(1, FOLDER_RANGE_BG[folder-1]);

bg = cv2.imread("./My_Flickr/images/" + str(folder) + "/" + str(file) + ".jpg")
#bg = cv2.resize(bg, (int(bg.shape[1] * 0.15),int(bg.shape[0] * 0.15)))
print(f"Background file: {folder}/{file}.jpg")

n_pics = int(input("Enter the number of stickers to be generated: "))

for n in range(n_pics):
	folder = randint(1,10)
	file = randint(0, FOLDER_RANGE[folder-1])
	img = cv2.imread("./Imagens_Teste/" + str(folder) + "/"+str(file)+".jpg")

	print(f"Sticker file {n}: {folder}/{file}.jpg")
	# Calcula a saliencia para a imagem lida 
	sal = cv2.saliency.StaticSaliencySpectralResidual_create()
	map = sal.computeSaliency(img)

	map = (map[1] * 255).astype("uint8")

	# Calcula o Threshold de Otsu sobre a saliencia da imagem
	thresh = cv2.threshold(map.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	#cv2.imshow('image antes', thresh)

	# Encontra os contornos da imagem
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	x, y = thresh.shape

	arr = np.zeros((x, y, 3), np.uint8)

	# Gera o 'sticker' iterando pelos contornos obtidos e recortando da imagem original
	for i in range(len(contours)):
		cnt = contours[i]
		if cv2.contourArea(cnt) > 0.05 * img.shape[0] * img.shape[1]:
			cv2.fillConvexPoly(arr, cnt, [255, 255, 255])
			
	#cv2.imshow('arr', arr)
	threshimg = np.array(img)
	
	# Arvore de decisao do quadrante sobre o qual o sticker sera colado
	# Visando evitar um numero alto demais de overlap
	if n % 4 == 0:
		
		randx = randint(low=0, high=(bg.shape[0] - threshimg.shape[0])//2)
		randy = randint(low=0, high=(bg.shape[1] - threshimg.shape[1])//2)
	elif n % 4 == 1:
		randx = randint(low=(bg.shape[0] - threshimg.shape[0])//2, high=bg.shape[0] - threshimg.shape[0])
		randy = randint(low=0, high=(bg.shape[1] - threshimg.shape[1])//2)
	elif n % 4 == 2:
		randx = randint(low=(bg.shape[0] - threshimg.shape[0])//2, high=bg.shape[0] - threshimg.shape[0])
		randy = randint(low=(bg.shape[1] - threshimg.shape[1])//2, high=bg.shape[1] - threshimg.shape[1])
	elif n % 4 == 3:
		randx = randint(low=0, high=(bg.shape[0] - threshimg.shape[0])//2)
		randy = randint(low=(bg.shape[1] - threshimg.shape[1])//2, high=bg.shape[1] - threshimg.shape[1])

	# Colagem per se do sticker sobre o background com interpolacao entre pixels do background e da imagem
	for i in range(randx, randx + img.shape[0]):
		for j in range(randy, randy + img.shape[1]):
			if arr[i - randx][j - randy][0] != 0 and arr[i - randx][j - randy][1] != 0 and arr[i - randx][j - randy][2] != 0:
				bg[i][j] = (3 * (threshimg[i - randx][j - randy].astype(float)) + (bg[i][j].astype(float))) // 4

	# Atualizacao do Background agora contendo o sticker
	bg = bg.astype(np.uint8)


cv2.imshow('collage', bg)
cv2.waitKey(0)