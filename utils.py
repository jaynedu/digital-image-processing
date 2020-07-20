# -*- coding: utf-8 -*-
# @Time    : 2020/6/22 22:25
# @Author  : Du Jing
# @FileName: utils.py
# @Usage   : 工具

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy


def bgr2rgb(src):
    """
    使用plt显示cv的图
    :param src:
    :return: dst
    """
    b, g, r = cv2.split(src)
    dst = cv2.merge([r, g, b])
    return dst


def maxpooling(src, ksize=2, strides=1):
    """
    最大池化
    :param src:
    :param ksize: 池化大小 default = 2
    :param strides: 步长 default = 1
    :return:
    """
    pool = np.zeros_like(src)
    for channel in range(src.shape[2]):
        height = 0
        for i in np.arange(0, src.shape[0], strides):
            width = 0
            for j in np.arange(0, src.shape[1], strides):
                pool[height, width, channel] = np.max(src[i: i + ksize, j:j + ksize, channel])
                width += 1
            height += 1
    return pool


def binarization(src, threshold: int):
    """
    二值化
    :param src:
    :param threshold:
    :return:
    """
    dst = np.array(copy.deepcopy(src))
    dst = np.where(dst < threshold, 0, 255)
    return dst

