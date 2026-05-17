#!/usr/bin/env python3

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def read(path):
	img = cv2.imread(path)

	cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)
	return img

def plot(img, title, save=os.getenv("SCRIPT_SAVE_IMG") is not None, show=False):
	if save:
		cv2.cvtColor(img, cv2.COLOR_RGB2BGR, dst=img)
		cv2.imwrite(f'Figūras/{ title }.jpg', img)
		return

	plt.figure()
	plt.imshow(img)

	plt.title(title)
	plt.axis(False)

	if show:
		plt.show()

def gaussian_blur(img, kernel=5):
	cv2.GaussianBlur(img, (kernel, kernel), 0, dst=img)
	return img


A = read('Bildes/Vilciens.jpg')
plot(A, 'Vilciens', save=False)

B = gaussian_blur(A)
plot(B, 'Vilciens', show=True)

