# -*- coding: utf-8 -*-
# @Time    : 2020/6/24 22:03
# @Author  : Du Jing
# @FileName: gabor.py
# @Usage   : Gabor

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
图像增强
反向热传导模型

1. Gabor滤波器可以很好的近似单细胞的感受野细胞（光强刺激下的传递函数），在提取目标的局部空间和频率域信息方面具有良好的特性。
2. 虽然Gabor小波本身不能构成正交基，但在特定参数下可构成紧框架。Gabor小波对于图像的边缘敏感，能够提供良好的方向选择和尺度选择特性，
   而且对于光照变化不敏感,能够提供对光照变化良好的适应性。-------Gabor小波被广泛应用于视觉信息理解
3. 二维Gabor小波变换是在时频域进行信号分析处理的重要工具，其变换系数有着良好的视觉特性和生物学背景。
    ------因此被广泛应用于图像处理、模式识别等领域。
4. 与传统的傅立叶变换相比，Gabor小波变换具有良好的时频局部化特性。
   即非常容易地调整Gabor滤波器的方向、基频带宽及中心频率从而能够最好的兼顾信号在时空域和频域中的分辨能力.
5. Gabor小波变换具有多分辨率特性即变焦能力。即采用多通道滤波技术，将一组具有不同时频域特性的Gabor小波应用于图像变换，
   每个通道都能够得到输入图像的某种局部特性，这样可以根据需要在不同粗细粒度上分析图像。
6. 在特征提取方面，Gabor小波变换与其它方法相比：一方面其处理的数据量较少，能满足系统的实时性要求；
   另一方面，小波变换对光照变化不敏感，且能容忍一定程度的图像旋转和变形，
   当采用基于欧氏距离进行识别时，特征模式与待测特征不需要严格的对应，故能提高系统的鲁棒性。
7. Gabor变换所采用的核（Kernels）与哺乳动物视觉皮层简单细胞2D感受野剖面（Profile）非常相似，
   具有优良的空间局部性和方向选择性，能够抓住图像局部区域内多个方向的空间频率（尺度）和局部性结构特征。
   这样，Gabor分解可以看作一个对方向和尺度敏感的有方向性的显微镜。
8. 二维Gabor函数也类似于增强边缘以及峰、谷、脊轮廓等底层图像特征，
   这相当于增强了被认为是面部关键部件的眼睛、鼻子、嘴巴等信息，同时也增强了诸于黑痣、酒窝、伤疤等局部特征，
   从而使得在保留总体人脸信息的同时增强局部特性成为可能。
   它的小波特性说明了Gabor滤波结果是描述图像局部灰度分布的有力工具,因此,可以使用Gabor滤波来抽取图像的纹理信息。
9. 由于Gabor特征具有良好的空间局部性和方向选择性，而且对光照、姿态具有一定的鲁棒性，因此在人脸识别中获得了成功的应用。
   然而，大部分基于Gabor特征的人脸识别算法中，只应用了Gabor幅值信息，而没有应用相位信息，
   主要原因是Gabor相位信息随着空间位置呈周期性变化，而幅值的变化相对平滑而稳定，幅值反映了图像的能量谱，
   Gabor幅值特征通常称为Gabor能量特征(Gabor Energy Features）。
   Gabor小波可像放大镜一样放大灰度的变化，人脸的一些关键功能区域(眼睛、鼻子、嘴、眉毛等)的局部特征被强化，从而有利于区分不同的人脸图像。
10. Gabor小波核函数具有与哺育动物大脑皮层简单细胞的二维反射区相同的特性，即具有较强的空间位置和方向选择性，
    并且能够捕捉对应于空间和频率的局部结构信息；Gabor滤波器对于图像的亮度和对比度变化以及人脸姿态变化具有较强的健壮性，
    并且它表达的是对人脸识别最为有用的局部特征。Gabor 小波是对高级脊椎动物视觉皮层中的神经元的良好逼近，是时域和频域精确度的一种折中。
"""

img = cv2.imread('src/lena.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sigma = 1
theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]
lamda = 2
psi = 0
gamma = 0.5
ksize = [7, 9, 11, 13, 15, 17]

"""Step 1 构建Gabor滤波器"""
def gabor_kernel(size, theta, sigma, lamda, psi, gamma=0.5):
    """
    构造gabor滤波器
    实部可以对图像进行平滑滤波，虚部可以用来边缘检测，实部和虚部相互正交
    :param x: 输入像素点x坐标 -> h
    :param y: 输入像素点y坐标 -> w
    :param size: gabor滤波器尺度
    :param theta: 方向，Gabor核函数图像的倾斜角度
    :param sigma: 带宽，高斯滤波器标准差
    :param lamda: 波长，直接影响滤波器的滤波尺度，通常大于等于2
    :param psi: 相位偏移，调谐函数的相位偏移，取值 -180 ~ 180
    :param gamma: 空间纵横比（长宽比），决定滤波器的形状椭圆率，取1时为圆形，通常取 0.5
    :return:
    """
    # (x, y) = np.meshgrid(np.arange(-(size-1)/2, (size+1)/2), np.arange(-(size-1)/2, (size+1)/2))
    (x, y) = np.meshgrid(np.arange(0, size), np.arange(0, size))
    _x = x*np.cos(theta) + y*np.sin(theta)  # x'
    _y = -x*np.sin(theta) + y*np.cos(theta)  # y'
    gb_real = np.exp(-(_x**2 + gamma**2 * _y**2)/(2 * sigma**2)) * np.cos(2*np.pi*_x/lamda + psi)
    gb_image = np.exp(-(_x**2 + gamma**2 * _y**2)/(2 * sigma**2)) * np.sin(2*np.pi*_x/lamda + psi)
    return gb_real, gb_image


def draw_gabor():
    plt.figure(figsize=(16, 16))
    n = 1
    for i, th in enumerate(theta):
        for j, size in enumerate(ksize):
            gabor_real, gabor_image = gabor_kernel(size, th, sigma, lamda, psi, gamma)
            plt.subplot(8, 6, n)
            plt.imshow(gabor_real)
            plt.subplot(8, 6, n+6)
            plt.imshow(gabor_image)
            n += 1
        n += 6
    plt.show()


if __name__ == '__main__':
    draw_gabor()



