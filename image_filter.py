# -*- coding: utf-8 -*-
# @Time    : 2020/6/23 10:50
# @Author  : Du Jing
# @FileName: image_filter.py
# @Usage   : 图像滤波

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""
图像能量大部分集中在幅度谱的低频和中频段，较高频段的信息经常被噪声淹没，因此一个能降低高频成分幅度的滤波器就能够减弱噪声的影响 
图像滤波的目的有两个:
    一是抽出对象的特征作为图像识别的特征模式;
    二是为适应图像处理的要求，消除图像数字化时所混入的噪声。
而对滤波处理的要求也有两条:
    一是不能损坏图像的轮廓及边缘等重要信息;
    二是使图像清晰视觉效果好。
平滑滤波是低频增强的空间域滤波技术。它的目的有：
    一是模糊；
    二是消除噪音。
空间域的平滑滤波一般采用简单平均法进行，求邻近像元点的平均亮度值。
邻域的大小与平滑的效果直接相关，邻域越大平滑的效果越好，
但邻域过大，平滑会使边缘信息损失的越大，从而使输出的图像变得模糊，因此需合理选择邻域的大小。
"""


def mean_kernel(size):
    """
    均值滤波核
    中心点的像素为域内所有像素点的平均值
    均值滤波本身存在着固有的缺陷，即它不能很好地保护图像细节，在图像去噪的同时也破坏了图像的细节部分，
    从而使图像变得模糊，不能很好地去除噪声点，特别是椒盐噪声
    :param size:
    :return:
    """
    kernel = np.ones((size, size))
    kernel = kernel / np.sum(kernel)
    return kernel


def gauss_kernel(size, sigma):
    """
    高斯核
    :param size: size = 2k+1
    :param sigma:
    :return:
    """
    kernel = np.zeros((size, size))
    center = (size - 1) // 2
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

    kernel = kernel / np.sum(kernel)
    return kernel


def log_kernel(size, sigma):
    return marr_kernel(size, sigma)


def marr_kernel(size, sigma):
    """
    LoG / Marr
    LoG边缘检测则是先进行高斯滤波再进行拉普拉斯算子检测，然后找过零点来确定边缘位置
    :param size: 通常为1+6sigma
    :param sigma:
    :return:
    """
    kernel = np.zeros((size, size))
    center = int((size - 1) // 2)
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = (-1 / (np.pi * sigma ** 4)) * (1 - (x ** 2 + y ** 2) / (2 * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)
    return kernel


def image_convolution(src, kernel):
    """
    使用核函数对图像卷积
    图像滤波的本质是使用核函数进行卷积
    :param src: 灰度图
    :param kernel:
    :return:
    """
    h, w = src.shape[:2]
    h_kernel, w_kernel = kernel.shape
    h_expand, w_expand = (h_kernel - 1) // 2, (w_kernel - 1) // 2
    h_conv, w_conv = int(h + h_expand * 2), int(w + w_expand * 2)

    dst = np.zeros_like(src)  # 输出结果
    conv = np.zeros((h_conv, w_conv))  # 用于卷积的矩阵
    conv[h_expand: h_expand + h, w_expand: w_expand + w] = src[:, :]
    for i in range(h_expand, h_expand + h):
        for j in range(w_expand, w_expand + w):
            dst[i - h_expand, j - w_expand] = np.sum(conv[i - h_expand: i + h_expand + 1, j - w_expand: j + w_expand + 1] * kernel)

    dst = cv2.convertScaleAbs(dst)
    return dst


def median_filter(src, sigma):
    """
    中值滤波 无核函数
    中心点像素为域内所有像素点的中值
    :param src: 灰度图
    :param sigma: 步长
    :return:
    """
    h, w = src.shape[:2]
    dst = np.zeros_like(src)
    for i in range(sigma // 2, h - sigma // 2):
        for j in range(sigma // 2, w - sigma // 2):
            area = []
            for k in range(-sigma//2, sigma//2+1):
                for m in range(-sigma//2, sigma//2+1):
                    area.append(src[i+k, j+m])
            np.sort(area, axis=None)
            dst[i, j] = np.median(area)
    return dst


def draw_kernel():
    """
    绘制三维图像

    常用的渐变cmap
    cmap = ['viridis', 'Spectral', 'rainbow', 'winter']
    :param kernel:
    :return:
    """
    ax = Axes3D(plt.figure())
    x = y = np.arange(-3, 3, 0.1)
    x, y = np.meshgrid(x, y)

    # 其他参数变量设置
    sigma = np.pi
    theta = np.pi/4
    lamda = 2
    psi = 0
    gamma = 0.5

    # 待绘制的函数
    z = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)  # 高斯函数
    z = (-1 / (np.pi * sigma ** 4)) * (1 - (x ** 2 + y ** 2) / (2 * sigma ** 2)) * np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2))  # LoG 高斯拉普拉斯算子

    _x = x * np.cos(theta) + y * np.sin(theta)  # x'
    _y = -x * np.sin(theta) + y * np.cos(theta)  # y'
    z = np.exp(-(_x**2 + gamma**2 * _y**2)/(2 * sigma**2)) * np.cos(2*np.pi*_x/lamda + psi)

    ax.plot_surface(x, y, z, cmap='viridis')
    plt.show()


if __name__ == '__main__':
    # 绘制核函数图像
    draw_kernel()

    # # 读取图片
    # img_gray = cv2.imread('src/bowl.tiff', cv2.IMREAD_GRAYSCALE)
    #
    # # 创建核
    # size = 15
    # sigma = 3
    # kernel_gauss = gauss_kernel(size, sigma)
    # kernel_mean = mean_kernel(size)
    # kernel_log = marr_kernel(size, sigma)
    #
    # # 卷积处理
    # img_gauss = image_convolution(img_gray, kernel_gauss)
    # img_mean = image_convolution(img_gray, kernel_mean)
    # img_median = median_filter(img_gray, sigma)
    # img_log = image_convolution(img_gray, kernel_log)
    #
    # # 结果汇总
    # img_show_list = [img_gray, img_gauss, img_mean, img_median, img_log]
    # title_list = ['raw', 'gauss', 'mean', 'median', 'LoG']
    # # 开始绘图
    # nrow, ncol = 2, 3
    # plt.figure()
    # for i, img in enumerate(img_show_list):
    #     plt.subplot(nrow, ncol, i + 1)
    #     plt.imshow(img, 'gray')
    #     plt.title(title_list[i])
    # plt.show()