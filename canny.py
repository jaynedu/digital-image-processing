# -*- coding: utf-8 -*-
# @Time    : 2020/6/24 15:04
# @Author  : Du Jing
# @FileName: canny.py
# @Usage   : Canny

import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
1.原始图像与高斯核卷积，获得稍模糊的图像，目的是降噪，因为导数对噪声敏感
2.使用一阶偏导算子sobel计算梯度
3.非极大值抑制，寻找像素点局部的最大值，目的是排除非边缘像素
4.双阈值法抑制假边缘，连接真边缘
    低于阈值1的像素点会被认为不是边缘；
    高于阈值2的像素点会被认为是边缘；
    在阈值1和阈值2之间的像素点,若与第2步得到的边缘像素点相邻，则被认为是边缘，否则被认为不是边缘。
"""

img = cv2.imread('src/lena.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

"""Step 1 Gaussian"""
from image_filter import gauss_kernel, image_convolution
kernel_size = 5
kernel_gauss = gauss_kernel(kernel_size, 1.5)
gaussian = image_convolution(img_gray, kernel_gauss)
fft_gaussian = np.fft.fft2(gaussian)

kernel_gauss_pad = np.pad(kernel_gauss, [[0, img_gray.shape[0]-kernel_size], [0, img_gray.shape[1]-kernel_size]])
fft_kernel = np.fft.fft2(kernel_gauss_pad)
fft_img_gray = np.fft.fft2(img_gray)
fft_gaussian = fft_img_gray * fft_kernel
ifft_gaussian_show = cv2.convertScaleAbs(abs(np.fft.ifft2(fft_gaussian)))

"""Step 2 Gradient"""
op_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
op_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

kernel_sobel_x = np.zeros_like(fft_gaussian, dtype=np.float32)
kernel_sobel_x[:op_x.shape[0], :op_x.shape[1]] = op_x
fft_kernel_sobel_x = np.fft.fft2(kernel_sobel_x)
kernel_sobel_y = np.zeros_like(fft_gaussian, dtype=np.float32)
kernel_sobel_y[:op_y.shape[0], :op_y.shape[1]] = op_y
fft_kernel_sobel_y = np.fft.fft2(kernel_sobel_y)
Gx = np.real(np.fft.ifft2(fft_kernel_sobel_x * fft_gaussian))
Gy = np.real(np.fft.ifft2(fft_kernel_sobel_y * fft_gaussian))
G = (Gx ** 2 + Gy ** 2) ** 0.5
Theta = np.arctan2(Gy, Gx) * 180 / np.pi

"""Step 3 Non-Max Suppression"""
local_maximum = np.zeros_like(G, dtype=np.float32)
h, w = G.shape[:2]
angle = abs(Theta)
for i in range(1, h-1):
    for j in range(1, w-1):
        # 0 degrees
        if (0<=angle[i, j]<22.5) or (157.5<=angle[i, j]<=180) :
            if (G[i, j] >= G[i, j+1]) and (G[i, j] >= G[i, j-1]):
                local_maximum[i, j] = G[i, j]
        # 45 degrees
        elif (22.5<=angle[i, j]<67.5):
            if (G[i, j] >= G[i-1, j+1]) and (G[i, j] >= G[i+1, j-1]):
                local_maximum[i, j] = G[i, j]
        # 90 degrees
        elif (67.5<=angle[i, j]<112.5):
            if (G[i, j] >= G[i-1, j]) and (G[i, j] >= G[i+1, j]):
                local_maximum[i, j] = G[i, j]
        # 135 degrees
        elif (112.5<=angle[i,j]<157.5):
            if (G[i, j] >= G[i-1, j-1]) and (G[i, j] >= G[i+1, j+1]):
                local_maximum[i, j] = G[i, j]

"""Step 4 Double Thresholding"""
threshold = np.zeros_like(local_maximum, np.float32)
strong = 1
weak = 0.5
maximum = np.max(local_maximum)
print("maximum:", maximum)
low = 0.1 * maximum
high = 0.25 * maximum
h, w = local_maximum.shape[:2]
for i in range(h):
    for j in range(w):
        if local_maximum[i, j] >= high:
            threshold[i, j] = strong
        elif local_maximum[i, j] >= low:
            threshold[i, j] = weak

"""Step 5 Tracking"""
h, w = threshold.shape[:2]
for i in range(h):
    for j in range(w):
        if threshold[i, j] == weak:
            try:
                if threshold[i, j+1] == strong or \
                    threshold[i, j-1] == strong or \
                    threshold[i+1, j] == strong or \
                    threshold[i-1, j] == strong or \
                    threshold[i+1, j+1] == strong or \
                    threshold[i+1, j-1] == strong or \
                    threshold[i-1, j+1] == strong or \
                    threshold[i-1, j-1] == strong:
                    threshold[i, j] = strong
                else:
                    threshold[i, j] = 0
            except IndexError:
                pass
threshold *= 255
edge = cv2.convertScaleAbs(threshold)

show_list = [img_gray, gaussian, ifft_gaussian_show, edge]
title_list = ['raw gray', 'gaussian', 'ifft_gaussian', 'edge']
plt.figure(figsize=(12, 3))
nrow, ncol = 1, 4
for i, img in enumerate(show_list):
    plt.subplot(nrow, ncol, i+1)
    plt.imshow(img, 'gray')
    plt.title(title_list[i])
plt.show()

