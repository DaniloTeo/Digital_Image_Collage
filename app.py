import cv2
from cv2 import saliency
import numpy as np
from numpy.random import randint
from scipy import stats
import random

# k-means para fazer colagem com objetos similares ou diferentes [dataset de stickers]

# Folder sizes for the Imagens_Teste/ folder
FOLDER_RANGE = [12, 18, 17, 4, 11, 30, 2, 6, 10, 0]

# Folder sizes for the My_Flickr/ folder
FOLDER_RANGE_BG = [388, 30, 357, 24, 390, 28]

class Cluster:
	def __init__(self, centroid, size):
		self.tam = 0
		self.array = np.zeros((size, centroid.shape[0]))
		self.centroid = centroid

	def add(self, img):
		self.array[self.tam] = img
		self.tam += 1

	def calc_centroid(self):
		dim = self.centroid.shape[0]
		summ = np.zeros(dim)

		for i in range(self.tam):
			for j in range(dim):
				summ[j] = summ[j] + self.array[i][j]

		self.centroid = summ/(self.tam+1)

	def return_images_array(self, sticker_list, arr_list):
		slist = []
		alist = []
		for i in range(self.tam):
			slist.append(sticker_list[int(self.array[i][3])])
			alist.append(arr_list[int(self.array[i][3])])
		return slist, alist

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

	print(f"Background file: {folder}/{file}.jpg")
	return bg

def get_sticker_list():
	sticker_list = []
	dataset = []
	arr_list = []
	count = 0
	for folder in range(1, 10):
		for file in range(FOLDER_RANGE[folder-1]):

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
			a = stats.mode(threshimg)[0][0][0]
			b = np.zeros(4)
			for i in range(3):
				b[i] = a[i]
			b[3] = count
			dataset.append(b)
			sticker_list.append(threshimg)
			arr_list.append(arr)
			count += 1

	return sticker_list, dataset, arr_list

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

def which_cluster(img, cluster, it, k):
	dist = np.zeros(k)

	for i in range(k):
		r = cluster[i].centroid[0]
		g = cluster[i].centroid[1]
		b = cluster[i].centroid[2]
		dist[i] = np.sqrt(((img[0]-r)**2) + ((img[1]-g)**2) + ((img[2]-b)**2))

	for i in range(it):
		indices = np.argwhere(dist == -1).flatten()
		dist = np.delete(dist, indices)
	
	w = np.argmin(dist)

	return w

def clear_clusters(cluster, k, size):
	for i in range(k):
		cluster[i].array = np.zeros((size, cluster[i].centroid.shape[0]))
	return cluster


def kmeans(ilist, k, n, seed):
	random.seed(seed)
	ids = np.sort(random.sample(range(0, len(ilist)), k))
	cluster = np.zeros(shape=k, dtype=object)

	for i in range(k):
		cluster[i] = Cluster(ilist[ids[i]], len(ilist))

	for i in range(n):
		for x in range(len(ilist)):
			w = which_cluster(ilist[x], cluster, 0, k)
			cluster[w].add(ilist[x])
		for j in range(k):
			if cluster[j].tam > 0:
				cluster[j].calc_centroid()
			if i < n-1:
				for l in range(k):
					cluster[l].tam = 0
	
	return cluster

def select_sticker(bg, sticker_list, arr_list, dataset, n_pics):
	# Criacao do vetor de clusters
	cluster = kmeans(dataset, k = 10, n = 3, seed = 30)

	for i in range(10):
		print(cluster[i].tam)

	# Identifica qual cluster o background mais se aproxima
	w = which_cluster(bg, cluster, 0, k = 10)

	slist, alist = cluster[w].return_images_array(sticker_list, arr_list)
	it = 1
	while len(alist) < n_pics:
		w = which_cluster(bg, cluster, it, k = 10)
		s, a = cluster[w].return_images_array(sticker_list, arr_list)
		slist.append(s)
		alist.append(a)

	return slist, alist

def collage(randx, randy, img, arr,bg):
	# Colagem per se do sticker sobre o background com interpolacao entre pixels do background e da imagem
	for i in range(randx, randx + img.shape[0]):
		for j in range(randy, randy + img.shape[1]):
			if arr[i - randx][j - randy][0] != 0 and arr[i - randx][j - randy][1] != 0 and arr[i - randx][j - randy][2] != 0:
				bg[i][j] = (3 * (img[i - randx][j - randy].astype(float)) + (bg[i][j].astype(float))) // 4
	return bg.astype(np.uint8)

def main():
	bg = get_background()
	#bg = cv2.resize(bg, (int(bg.shape[1] * 0.15),int(bg.shape[0] * 0.15)))

	n_pics = int(input("Enter the number of stickers to be generated: "))

	sticker_list, dataset, arr_list = get_sticker_list()

	sticker_selected, arr_selected = select_sticker(stats.mode(bg)[0][0][0], sticker_list, arr_list, dataset, n_pics)

	for i in range(n_pics):

		# Get the position it'll be pasted on
		randx, randy = position_sticker(i, sticker_selected[i], bg)

		# Atualizacao do Background agora contendo o sticker
		bg = collage(randx, randy, sticker_selected[i], arr_selected[i], bg)

	cv2.imshow('collage', bg)
	cv2.waitKey(0)

if __name__ == '__main__':
	main()