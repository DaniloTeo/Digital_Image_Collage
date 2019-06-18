import cv2
from cv2 import saliency
import numpy as np
from numpy.random import randint
import operator

# k-means para fazer colagem com objetos similares ou diferentes [dataset de stickers]

FOLDER_RANGE = [12, 18, 17, 4, 11, 30, 2, 6, 10, 0]
FOLDER_RANGE_BG = [387, 30, 357, 24, 390, 29]

# folder = randint(1,7);print(folder)
# file = randint(1, FOLDER_RANGE_BG[folder-1]);print(file)
#bg = cv2.imread('bg.jpg')
aux = []
for folder in range(1, 7):
	for file in range(1, (FOLDER_RANGE_BG[folder-1] + 1)):
		bg = cv2.imread("./My_Flickr/images/" + str(folder) + "/" + str(file) + ".jpg")
		#bg = cv2.resize(bg, (int(bg.shape[1] * 0.15),int(bg.shape[0] * 0.15)))
		d = {"folder": folder, "file": file, "shape": bg.shape}
		aux.append(d)

l = sorted(aux, key=operator.itemgetter('shape'), reverse=True)

for i in l:
	print(i)
