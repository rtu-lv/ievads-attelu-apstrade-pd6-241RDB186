#!/usr/bin/env python3

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def read(path):
	img = cv2.imread(path)

	cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)
	return img

def plot(img, title=None, save=os.getenv("SCRIPT_SAVE_IMG") is not None, show=False):
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


def gaussian_blur(img, *, kernel):
	cv2.GaussianBlur(img, (kernel, kernel), 0, dst=img)
	return img

def otsu_thresh(img):
	grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	_, mask = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return mask

def morph_open(mask, *, kernel):
	struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
	return cv2.morphologyEx(mask, cv2.MORPH_OPEN, struct)


A = read('Bildes/Vilciens.jpg')
plot(A, 'Vilciens', save=False)

B = otsu_thresh(A)
plot(B)

B = morph_open(B, kernel=5)
plot(B, show=True)

