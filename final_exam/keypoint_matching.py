# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 13:20
# @Author  : Du Jing
# @FileName: keypoint_matching.py
# @Usage   : 可否直接采用SIFT之类的算法，实现两个图像关键特征的匹配？路面能匹配起来吗？

import cv2
import numpy as np

"""分块选择不同输入"""
# # 原始输入
# img1 = cv2.imread('road-1.jpg', 0)
# img2 = cv2.imread('road-2.jpg', 0)


# 自适应直方图均衡化
img1 = cv2.imread('road-1.jpg', 0)
img2 = cv2.imread('road-2.jpg', 0)
clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(7, 7))
img1 = clahe.apply(img1)
img2 = clahe.apply(img2)

# # 灰度世界
# def gray_world(img):
#     b, g, r = cv2.split(img)
#     avgb = np.mean(b)
#     avgg = np.mean(g)
#     avgr = np.mean(r)
#     gray = np.mean([avgb, avgg, avgr])
#     kb, kg, kr = gray/avgb, gray/avgg, gray/avgr
#     dst = cv2.merge([b*kb, g*kg, r*kr])
#     return cv2.convertScaleAbs(dst)
# img1 = cv2.imread('road-1.jpg')
# img2 = cv2.imread('road-2.jpg')
# img1 = gray_world(img1)
# img2 = gray_world(img2)

# 计算关键点
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


# 使用BF匹配
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
newimg = cv2.drawMatches(img1, kp1, img2, kp2, matches[:200], img2, flags=2)
cv2.imwrite('sift_matching_result.png', newimg)
