# -*- coding: utf-8 -*-
# @Time    : 2020/6/23 20:40
# @Author  : Du Jing
# @FileName: dct.py
# @Usage   : DCT与图像压缩

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size=14)
np.set_printoptions(threshold=np.inf)

"""
使用DCT的方式进行图像压缩：
    1 对图像做DCT
    2 对DCT的系数每个维度取前size个值
    3 对取得的小DCT系数做IDCT
    主要思路即仅小部分DCT系数就可重构图像
"""


def direct_dct(src):
    """
    直接DCT 按照定义
    :param src: 灰度图
    :return:
    """
    N = src.shape[0]
    alpha = np.sqrt(2/N) * np.ones((N, 1), dtype=np.float32)
    alpha[0] = np.sqrt(1/N)
    coef = np.ones_like(src, dtype=np.float32)
    for i in range(N):
        for j in range(N):
            coef[i, j] = alpha[i] * np.cos((j + 0.5) * i * np.pi / N)
    dst = np.matmul(np.matmul(coef, src), np.transpose(coef))
    return dst


def block_dct(src, n_block=8):
    """
    分块DCT
    反变换时需要分块反变换
    :param src: size为2的n次方（256，512）
    :return:
    """
    size = src.shape[0] // n_block
    dst = np.ones_like(src, dtype=np.float32)
    for i in range(n_block):
        for j in range(n_block):
            dst[i*size: (i+1)*size, j*size: (j+1)*size] = direct_dct(src[i*size: (i+1)*size, j*size: (j+1)*size])
    return dst


def direct_idct(coef):
    return cv2.idct(coef)


def block_idct(coef, n_block=8):
    size = coef.shape[0] // n_block
    idct = np.ones_like(coef, dtype=np.float32)
    for i in range(n_block):
        for j in range(n_block):
            idct[i*size: (i+1)*size, j*size: (j+1)*size] = cv2.idct(coef[i*size: (i+1)*size, j*size: (j+1)*size])
    return idct


if __name__ == '__main__':
    # 读取图像
    img_gray = cv2.imread('src/lena.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_gray = cv2.resize(img_gray, (512, 512))

    # 获取DCT系数
    coef_direct_dct = direct_dct(img_gray)
    coef_direct_dct_show = cv2.convertScaleAbs(coef_direct_dct)
    coef_block_dct = block_dct(img_gray, 8)
    coef_block_dct_show = cv2.convertScaleAbs(coef_block_dct)

    # 图像重构
    img_direct_dct = direct_idct(coef_direct_dct)
    img_direct_dct_show = cv2.convertScaleAbs(img_direct_dct)
    img_block_dct = block_idct(coef_block_dct, 8)
    img_block_dct_show = cv2.convertScaleAbs(img_block_dct)

    # 结果展示
    coef_show_list = [img_gray, coef_direct_dct_show, coef_block_dct_show]
    img_show_list = [img_gray, img_direct_dct_show, img_block_dct_show]
    title_list = ['raw', 'direct dct', 'block dct']
    nrow, ncol = 2, 3
    plt.figure()
    for i, img in enumerate(coef_show_list):
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(img, 'gray')
        plt.title(title_list[i])
        plt.subplot(nrow, ncol, i+ncol+1)
        plt.imshow(img_show_list[i], 'gray')
    plt.show()


