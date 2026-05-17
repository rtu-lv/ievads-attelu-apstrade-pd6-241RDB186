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


def gaussian_blur(img, *, kernel):
	cv2.GaussianBlur(img, (kernel, kernel), 0, dst=img)
	return img

def greyscale(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def log_correction(img, *, norm):
	data = np.float32(img) / 255
	data[:] = np.clip(norm * np.log(1 + data), 0, 1)

	img[:] = np.uint8(data * 255)
	return img

def otsu_thresh(grey):
	cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, dst=grey)
	return grey

def morph_open(grey, *, kernel, shape=cv2.MORPH_ELLIPSE):
	struct = cv2.getStructuringElement(shape, (kernel, kernel))

	cv2.morphologyEx(grey, cv2.MORPH_OPEN, struct, dst=grey)
	return grey

def as_mask(grey):
	mask = grey == 255
	return mask[:, :, np.newaxis]


def portrait_effect(img, *, blur=21):
	blur = gaussian_blur(img.copy(), kernel=blur)

	grey = greyscale(img)
	log_correction(grey, norm=10)

	otsu_thresh(grey)
	morph_open(grey, kernel=20)

	mask = as_mask(grey)
	img[:] = np.where(mask, blur, img)
	return img


A = read('Bildes/Vilciens.jpg')
plot(A, 'Vilciens', save=False)

portrait_effect(A)
plot(A, 'Vilciens - Portreta efekts', show=True)

