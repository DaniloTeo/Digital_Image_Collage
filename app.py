#Project - Digital Image Collage
#Danilo da Costa Telles Téo - 9293626
#Rodrigo Valim Maciel - 9278149
#Victor Roberti Camolesi - 9791239

import cv2
from cv2 import saliency
import numpy as np
from numpy.random import randint
from scipy import stats
from sklearn.cluster import KMeans

# Folder sizes for the Imagens_Teste/ folder
FOLDER_RANGE = [12, 18, 17, 4, 11, 30, 2, 6, 10, 0]

# Folder sizes for the My_Flickr/ folder
FOLDER_RANGE_BG = [388, 30, 357, 24, 390, 28]
# Calculate the mode of each image in the sticker list, considering only the selected pixels (in the get_sticker_list method)
def modas(sticker_list, arr_list):
	modas = []	
	for i in range(len(sticker_list)):
		aux_sticker = np.reshape(sticker_list[i], (sticker_list[i].shape[0] * sticker_list[i].shape[1], sticker_list[i].shape[2]))
		aux_arr = np.reshape(arr_list[i], (arr_list[i].shape[0] * arr_list[i].shape[1], arr_list[i].shape[2]))
		a = np.where(aux_arr == [255, 255, 255])[0]
		moda = stats.mode(aux_sticker[a], axis = 0)[0][0]
		modas.append(moda)
	return modas

# Use the KMeans algorithm to classify each sticker and the background, based on their modes.
# Then the index of the stickers with the same classification as the background is returned.
def clusters(bg, sticker_list, arr_list, moda_list, n):
	bg_mode = stats.mode(bg)[0][0][0]
	km = KMeans(n_clusters = n).fit(moda_list)
	bg_cluster = km.predict([bg_mode])
	return np.where(km.labels_ == bg_cluster)[0]

# Return random background from the background folder
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

# Returns a list of images for stickers (not cut at all) and a list of borders for these images. It omputes the salience and
# then the threshold for the images to bet put in the sticker_list. Then the border is found by finding the countours and filling the shape
# inside the countour with white. These calculations are used with OpenCV functions.
def get_sticker_list():
	sticker_list = []
	arr_list = []

	for folder in range(1, 10):
		for file in range(FOLDER_RANGE[folder-1]):

			img = cv2.imread("./Imagens_Teste/" + str(folder) + "/"+str(file)+".jpg")
		
			# Computes salience for input image 
			sal = cv2.saliency.StaticSaliencySpectralResidual_create()
			map = sal.computeSaliency(img)
			map = (map[1] * 255).astype("uint8")

			# Computes Otsu Threshold over the saliented image
			thresh = cv2.threshold(map.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

			# Find the countours of the image
			contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

			x, y = thresh.shape
			arr = np.zeros((x, y, 3), np.uint8)

			# Fill the shape delimited by the countours with white 
			for i in range(len(contours)):
				cnt = contours[i]
				if cv2.contourArea(cnt) > 0.05 * img.shape[0] * img.shape[1]:
					cv2.fillConvexPoly(arr, cnt, [255, 255, 255])
					
			threshimg = np.array(img)
			
			sticker_list.append(threshimg)
			arr_list.append(arr)

	return sticker_list, arr_list

# Returns the position the sticker will be pasted on the background
def position_sticker(n, sticker, bg):
	size_x = bg.shape[0] - sticker.shape[0]
	size_y = bg.shape[1] - sticker.shape[1]

	# Decides which quadrant the sticker will be put (4 quadrants) 
	# This has the ojective to reduce overllaping
	if n % 4 == 0:
		randx = randint(low=0, high=(size_x)//2)
		randy = randint(low=0, high=(size_y)//2)
	
	elif n % 4 == 1:
		randx = randint(low=(size_x)//2, high=size_x)
		randy = randint(low=0, high=(size_y)//2)
	
	elif n % 4 == 2:
		randx = randint(low=(size_x)//2, high=size_x)
		randy = randint(low=(size_y)//2, high=size_y)
	
	elif n % 4 == 3:
		randx = randint(low=0, high=(size_x)//2)
		randy = randint(low=(size_y)//2, high=size_y)

	return randx, randy

# Put the sticker on the background following the limits from the countour
# This also makes an interpolation of the sticker with the background
def collage(randx, randy, img, arr,bg):
	for i in range(randx, randx + img.shape[0]):
		for j in range(randy, randy + img.shape[1]):
			if arr[i - randx][j - randy][0] != 0 and arr[i - randx][j - randy][1] != 0 and arr[i - randx][j - randy][2] != 0:
				bg[i][j] = (3 * (img[i - randx][j - randy].astype(float)) + (bg[i][j].astype(float))) // 4
	return bg.astype(np.uint8)

def main():
	# Gets the background the stickers will be put on
	bg = get_background()
	
	# Returns list of stickers and countours
	sticker_list, arr_list = get_sticker_list()

	# Number of clusters to be used in the KMeans
	n_clusters = int(input("Insira o numero  de clusters para o KMeans: "))

	# Finds the mode of each sticker and return a list of these modes
	moda_list = modas(sticker_list, arr_list)

	# Return indexes of images to be used on the collage
	sticker_index = clusters(bg, sticker_list, arr_list, moda_list, n_clusters)
	
	for i in range(len(sticker_index)):
		# Get the position the sticker will be pasted on
		randx, randy = position_sticker(i, np.asarray(sticker_list[sticker_index[i]]), bg)

		# Update of background now containing the sticker
		bg = collage(randx, randy, np.asarray(sticker_list[sticker_index[i]]), arr_list[sticker_index[i]], bg)

	cv2.imshow('collage', bg)
	cv2.waitKey(0)

if __name__ == '__main__':
	main()