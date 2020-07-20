# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 13:03
# @Author  : Du Jing
# @FileName: edge_matching.py
# @Usage   : 可否直接采用边缘检测算法，实现两个图像边缘特征之间的匹配？路面能匹配起来吗？

import cv2
import numpy as np
import matplotlib.pyplot as plt

road1 = cv2.imread('road-1.jpg')
road2 = cv2.imread('road-2.jpg')

road1_gray = cv2.cvtColor(road1, cv2.COLOR_BGR2GRAY)
road2_gray = cv2.cvtColor(road2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# sobel边缘检测
sobel_1 = cv2.Sobel(road1_gray, cv2.CV_8U, 1, 1)
sobel_2 = cv2.Sobel(road2_gray, cv2.CV_8U, 1, 1)
sobel_matching = cv2.matchShapes(sobel_1, sobel_2, cv2.CHAIN_APPROX_SIMPLE, 0)
kp1 = orb.detect(sobel_1, None)
kp2 = orb.detect(sobel_2, None)
kp1, des1 = orb.compute(sobel_1, kp1)
kp2, des2 = orb.compute(sobel_2, kp2)
# 特征点匹配
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
newimg = cv2.drawMatches(road1, kp1, road2, kp2, matches[:50], road2, flags=2)
cv2.imwrite('sobel_matching_result.png', newimg)

# Canny边缘检测
max1 = np.max(road1_gray)
max2 = np.max(road2_gray)
maximum = max1 if max1 >= max2 else max2
canny_1 = cv2.Canny(road1_gray, 0.5*max1, 0.9*max1)
canny_2 = cv2.Canny(road2_gray, 0.5*max2, 0.9*max2)
canny_matching = cv2.matchShapes(canny_1, canny_2, cv2.CHAIN_APPROX_SIMPLE, 0)
kp1 = orb.detect(canny_1, None)
kp2 = orb.detect(canny_2, None)
kp1, des1 = orb.compute(canny_1, kp1)
kp2, des2 = orb.compute(canny_2, kp2)
# 特征点匹配
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
newimg = cv2.drawMatches(road1, kp1, road2, kp2, matches[:50], road2, flags=2)
cv2.imwrite('canny_matching_result.png', newimg)

show_list = [road1_gray, road2_gray, sobel_1, sobel_2, canny_1, canny_2]
title_list = ['road1', 'road2', 'sobel_1', 'sobel_2', 'canny_1', 'canny_2']
plt.figure(figsize=(8, 12))
nrow, ncol = 3, 2
for i, img in enumerate(show_list):
    plt.subplot(nrow, ncol, i+1)
    plt.imshow(img, 'gray')
    plt.title(title_list[i])
plt.savefig('edge_detection_result.png')
plt.show()

