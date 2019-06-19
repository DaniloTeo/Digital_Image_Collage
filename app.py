import cv2
from cv2 import saliency
import numpy as np
from numpy.random import randint
from scipy import stat

# k-means para fazer colagem com objetos similares ou diferentes [dataset de stickers]

# Folder sizes for the Imagens_Teste/ folder
FOLDER_RANGE = [12, 18, 17, 4, 11, 30, 2, 6, 10, 0]

# Folder sizes for the My_Flickr/ folder
FOLDER_RANGE_BG = [388, 30, 357, 24, 390, 28]


#-----------------------------------------------------------------------------------------
def get_background():
	# Pick the folder and the file randomly
	folder = randint(1,7);
	file = randint(1, FOLDER_RANGE_BG[folder-1]);
	
	# Read the file and return
	bg = cv2.imread("./My_Flickr/images/" + str(folder) + "/" + str(file) + ".jpg")
	
	# Guarantee the dimensions for the backgrounds will be suitable for use
	while (bg.shape[0] < 530) or (bg.shape[1] < 806):
		folder = randint(1,7);
		file = randint(1, FOLDER_RANGE_BG[folder-1]);
		bg = cv2.imread("./My_Flickr/images/" + str(folder) + "/" + str(file) + ".jpg")


	return bg

def get_sticker_list():
	sticker_list = []
	dataset = []
	arr_list = []

	for folder in range(1, 10):
		for file in range(0, FOLDER_RANGE[folder-1])

			img = cv2.imread("./Imagens_Teste/" + str(folder) + "/"+str(file)+".jpg")
			#print(f"Sticker file {n}: {folder}/{file}.jpg")
		
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
			
			# Guarda a Moda das imagens para o K-Means
			dataset.append(stats.mode(threshimg)[0][0])
			sticker_list.append(threshimg)
			arr_list.append(arr)

	return sticker_list, dataset

def position_sticker(n, sticker, bg):
	# Arvore de decisao do quadrante sobre o qual o sticker sera colado
	# Visando evitar um numero alto demais de overlap
	if n % 4 == 0:
		randx = randint(low=0, high=(bg.shape[0] - sticker.shape[0])//2)
		randy = randint(low=0, high=(bg.shape[1] - sticker.shape[1])//2)
	
	elif n % 4 == 1:
		randx = randint(low=(bg.shape[0] - sticker.shape[0])//2, high=bg.shape[0] - sticker.shape[0])
		randy = randint(low=0, high=(bg.shape[1] - sticker.shape[1])//2)
	
	elif n % 4 == 2:
		randx = randint(low=(bg.shape[0] - sticker.shape[0])//2, high=bg.shape[0] - sticker.shape[0])
		randy = randint(low=(bg.shape[1] - sticker.shape[1])//2, high=bg.shape[1] - sticker.shape[1])
	
	elif n % 4 == 3:
		randx = randint(low=0, high=(bg.shape[0] - sticker.shape[0])//2)
		randy = randint(low=(bg.shape[1] - sticker.shape[1])//2, high=bg.shape[1] - sticker.shape[1])

	return randx, randy

def select_sticker():
	# K-Means vem aqui
	# Retorna tanto o elemento de contorno arr quanto o sticker
	pass

def collage(randx, randy, img, arr,bg):
	# Colagem per se do sticker sobre o background com interpolacao entre pixels do background e da imagem
	for i in range(randx, randx + img.shape[0]):
		for j in range(randy, randy + img.shape[1]):
			if arr[i - randx][j - randy][0] != 0 and arr[i - randx][j - randy][1] != 0 and arr[i - randx][j - randy][2] != 0:
				bg[i][j] = (3 * (img[i - randx][j - randy].astype(float)) + (bg[i][j].astype(float))) // 4
	return bg.astype(np.uint8)
#----------------------------------------------------------------------------------------

bg = get_background()
print(f"Background file: {folder}/{file}.jpg")

#bg = cv2.resize(bg, (int(bg.shape[1] * 0.15),int(bg.shape[0] * 0.15)))

n_pics = int(input("Enter the number of stickers to be generated: "))

sticker_list, dataset = get_sticker_list()
dataset.append(stats.mode(np.asarray(bg))[0][0])

for i in range(n_pics);
	# Get the sticker to be pasted on the background
	sticker, arr = select_sticker()

	# Get the position it'll be pasted on
	randx, randy = position_sticker(i, sticker, bg)

	# Atualizacao do Background agora contendo o sticker
	bg = collage(randx, randy, sticker, arr, bg)
	


cv2.imshow('collage', bg)
cv2.waitKey(0)