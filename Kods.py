#!/usr/bin/env python3

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def read(path):
	img = cv2.imread(path)

	cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)
	return img

def plot(img, title, *, save=os.getenv("SCRIPT_SAVE_IMG") is not None, show=False):
	if save:
		cv2.cvtColor(img, cv2.COLOR_RGB2BGR, dst=img)
		cv2.imwrite(f'Figūras/{ title }.jpg', img)
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


def improve_contrast(img, *, cluster=5, blur=9, iter=10):
	orig = img.copy()

	clust = k_means(img, k=cluster, iter=iter)
	blur = gaussian_blur(clust.copy(), kernel=blur)
	norm = np.float32(blur) / 255

	log_correction(img, norm=1+norm)

	return (img, blur, clust, orig)


A = read('Bildes/Vilciens.jpg')
plot(A, 'Vilciens', save=False)

improve_contrast(A)
plot(A, 'Vilciens - Uzlabots kontrasts', show=True)

