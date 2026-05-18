#!/usr/bin/env python3

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def read(path):
	img = cv2.imread(path)

	cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)
	return img

def plot(img, title, *, save=os.getenv('SCRIPT_SAVE_IMG') is not None, show=False):
	if save:
		cv2.cvtColor(img, cv2.COLOR_RGB2BGR, dst=img)
		cv2.imwrite(f'Figūras/{title}.jpg', img)
		return

	plt.figure()
	plt.imshow(img, cmap='grey')

	plt.title(title)
	plt.axis(False)

	if show:
		plt.show()


def k_means(img, *, k, iter, eps=0):
	data = np.float32(img.reshape(-1, 3))

	criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, iter, eps)
	origin = cv2.KMEANS_PP_CENTERS

	_, labels, centers = cv2.kmeans(data, k, None, criteria, iter, origin)

	labels = labels.flat
	centers = np.uint8(centers)

	grey = centers[labels]
	return grey.reshape(img.shape)

def gaussian_blur(img, *, kernel):
	cv2.GaussianBlur(img, (kernel, kernel), 0, dst=img)
	return img

def log_correction(img, *, norm):
	data = np.float32(img) / 255
	data[:] = np.clip(norm * np.log(1 + data), 0, 1)

	img[:] = np.uint8(data * 255)
	return img


def improve_contrast(img, *, norm, cluster=5, blur=9, iter=10):
	orig = img.copy()

	clust = k_means(img, k=cluster, iter=iter)
	blur = gaussian_blur(clust.copy(), kernel=blur)
	map = np.float32(blur) / 255

	log_correction(img, norm=1+(norm-1)*map)

	return (img, blur, clust, orig)


img = read(f'Bildes/Darbs.jpg')

img, blur, clust, orig = improve_contrast(img, norm=2)
plot(orig, 'Darbs', save=False)
plot(clust, 'Darbs - K-Means')
plot(blur, 'Darbs - K-Means Gausa filtrs')
plot(img, 'Darbs - Klasterētā logaritmiskā korekcija')

log = log_correction(orig, norm=2)
plot(log, 'Darbs - Globālā logaritmiskā korekcija', show=True)


img = read(f'Bildes/Kaķis.jpg')

img, blur, clust, orig = improve_contrast(img, norm=2)
plot(orig, 'Kaķis', save=False)
plot(clust, 'Kaķis - K-Means')
plot(blur, 'Kaķis - K-Means Gausa filtrs')
plot(img, 'Kaķis - Klasterētā logaritmiskā korekcija')

log = log_correction(orig, norm=2)
plot(log, 'Kaķis - Globālā logaritmiskā korekcija', show=True)


img = read(f'Bildes/Mašīna.jpg')

img, blur, clust, orig = improve_contrast(img, norm=1.5)
plot(orig, 'Mašīna', save=False)
plot(clust, 'Mašīna - K-Means')
plot(blur, 'Mašīna - K-Means Gausa filtrs')
plot(img, 'Mašīna - Klasterētā logaritmiskā korekcija')

log = log_correction(orig, norm=1.5)
plot(log, 'Mašīna - Globālā logaritmiskā korekcija', show=True)


img = read(f'Bildes/Telefona kabīne.jpg')

img, blur, clust, orig = improve_contrast(img, norm=1.5)
plot(orig, 'Telefona kabīne', save=False)
plot(clust, 'Telefona kabīne - K-Means')
plot(blur, 'Telefona kabīne - K-Means Gausa filtrs')
plot(img, 'Telefona kabīne - Klasterētā logaritmiskā korekcija')

log = log_correction(orig, norm=1.5)
plot(log, 'Telefona kabīne - Globālā logaritmiskā korekcija', show=True)


img = read(f'Bildes/Vilciens.jpg')

img, blur, clust, orig = improve_contrast(img, norm=2)
plot(orig, 'Vilciens', save=False)
plot(clust, 'Vilciens - K-Means')
plot(blur, 'Vilciens - K-Means Gausa filtrs')
plot(img, 'Vilciens - Klasterētā logaritmiskā korekcija')

log = log_correction(orig, norm=2)
plot(log, 'Vilciens - Globālā logaritmiskā korekcija', show=True)


img = read(f'Bildes/Vīrietis.jpg')

img, blur, clust, orig = improve_contrast(img, norm=2)
plot(orig, 'Vīrietis', save=False)
plot(clust, 'Vīrietis - K-Means')
plot(blur, 'Vīrietis - K-Means Gausa filtrs')
plot(img, 'Vīrietis - Klasterētā logaritmiskā korekcija')

log = log_correction(orig, norm=2)
plot(log, 'Vīrietis - Globālā logaritmiskā korekcija', show=True)

