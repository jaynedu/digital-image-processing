# -*- coding: utf-8 -*-
# @Time    : 2020/6/22 22:24
# @Author  : Du Jing
# @FileName: edge_matching.py
# @Usage   : 边缘检测

import cv2
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

""" 
边缘检测是图像处理与计算机视觉中极为重要的一种分析图像的方法，
边缘检测的目的就是找到图像中亮度变化剧烈的像素点构成的集合，表现出来往往是轮廓。
边缘即是图像的一阶导数局部最大值的地方，那么也意味着该点的二阶导数为零
如果图像中边缘能够精确的测量和定位，那么就意味着实际的物体能够被定位和测量，包括物体的面积、物体的直径、物体的形状等就能被测量。
在对现实世界的图像采集中，有下面4种情况会表现在图像中时形成一个边缘。
    深度的不连续（物体处在不同的物平面上）；
    表面方向的不连续（如正方体的不同的两个面）；
    物体材料不同（这样会导致光的反射系数不同）；
    场景中光照不同（如被树萌投向的地面）；

微分算子
    一阶微分算子
        利用图像在边缘处的阶跃性，即图像梯度在边缘取得极大值的特性进行边缘检测;
        梯度的方向提供了边缘的趋势信息，因为梯度方向始终是垂直于边缘方向，梯度的模值大小提供了边缘的强度信息。

        Roberts
            Roberts边缘算子是一个2x2的模板，采用的是对角方向相邻的两个像素之差。
            对具有陡峭的低噪声的图像处理效果较好，提取的边缘较粗，边缘定位不是很准确。

        Sobel
            Sobel也是用周围8个像素来估计中心像素的梯度，但是Sobel算子认为靠近中心像素的点应该给予更高的权重，
            所以Sobel算子把与中心像素4邻接的像素的权重设置为2或-2
            对灰度渐变和噪声较多的图像处理效果比较好，对边缘定位比较准确

        Prewitt
            Prewitt利用周围邻域8个点的灰度值来估计中心的梯度
            对灰度渐变和噪声较多的图像处理效果比较好。

    二阶微分算子
        二阶微分边缘检测算子就是利用图像在边缘处的阶跃性导致图像二阶微分在边缘处出现零值这一特性进行边缘检测的

        Laplacian
            1）用Laplace核与图像进行卷积；
            2）对卷积后的图像，取得那些卷积结果为0的点。
            对图像中的阶跃性边缘点定位准确，对噪声非常敏感，丢失一部分边缘的方向信息，造成一些不连续的检测边缘。
            缺点是对噪声十分敏感，同时也没有能够提供边缘的方向信息

        Marr/LoG
            为了实现对噪声的抑制，Marr等提出了LOG的方法。
            首先图像要进行低通滤波，LOG采用了高斯函数作为低通滤波器。
            σ决定了对图像的平滑程度。高斯函数生成的滤波模板尺寸一般设定为6σ+1（加1是为了使滤波器的尺寸为奇数）
            使用高斯函数对图像进行滤波并对图像滤波结果进行二阶微分运算的过程，可以转换为先对高斯函数进行二阶微分，
            再利用高斯函数的二阶微分结果对图像进行卷积运算
            经常出现在双边缘像素边界，而且该检测方法对噪声比较敏感，所以很少用LoG算子检测边缘，而是用来判断边缘像素是位于图像的明区还是暗区
非微分算子 
    Canny
        该方法不容易受到噪声的干扰，能够检测到真正的弱边缘。
        在edge函数中，最有效的边缘检测是Canny方法，其优点在于使用两种不同的阈值分别检测强边缘和弱边缘，
        并且仅当弱边缘与强边缘相连时才将弱边缘包含在输出图像中。因此该方法不容易被噪声填充，更容易检测出真正的弱边缘。

        在一阶微分算子的基础上，增加了非最大值抑制和双阈值两项改进。
        利用非极大值抑制不仅可以有效地抑制多响应边缘，而且还可以提高边缘的定位精度；
        利用双阈值可以有效减少边缘的漏检率。
        1)去噪声；
        2)计算梯度与方向角；
        3)非最大值抑制；
        4)滞后阈值化；
        其中前两步很简单，先用一个高斯滤波器对图像进行滤波，然后用Sobel水平和竖直检测子与图像卷积，来计算梯度和方向角。
        非极大值抑制
            寻找像素点局部最大值，将非极大值点所对应的灰度值置为0，这样可以剔除掉一大部分非边缘的点。
        滞后阈值化
            由于噪声的影响，经常会在本应该连续的边缘出现断裂的问题。
            滞后阈值化设定两个阈值：一个为高阈值Th ，一个为低阈值Tl。
            如果任何像素边缘算子的影响超过高阈值，将这些像素标记为边缘；
            响应超过低阈值（高低阈值之间）的像素，如果与已经标记为边缘的像素4-邻接或8-邻接，则将这些像素也标记为边缘。
"""


def robert(src):
    """
    [[-1,-1],[1,1]]
    :param src: 灰度图
    :return:
    """
    op = np.array([[-1, -1], [1, 1]])
    dst = np.zeros_like(src, dtype=np.float32)
    h, w = src.shape[:2]
    for i in range(h):
        for j in range(w):
            if ((j + 2) <= w) and ((i + 2) <= h):
                point = src[i: i + 2, j: j + 2] * op
                dst[i, j] = abs(point.sum())
    dst = cv2.convertScaleAbs(dst)
    return dst


def sobel(src):
    """
    x方向与y方向算子不同，需要分开计算
    :param src:
    :return:
    """
    h, w = src.shape[:2]
    dst_x = np.zeros((h, w), dtype=np.float32)
    dst_y = np.zeros((h, w), dtype=np.float32)
    dst = np.zeros((h, w), dtype=np.float32)
    op_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    op_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    for i in range(h - 2):
        for j in range(w - 2):
            dst_x[i + 1, j + 1] = abs(np.sum(src[i: i + 3, j: j + 3] * op_x))
            dst_y[i + 1, j + 1] = abs(np.sum(src[i: i + 3, j: j + 3] * op_y))
            dst[i + 1, j + 1] = (dst_x[i + 1, j + 1] * dst_x[i + 1, j + 1]
                                 + dst_y[i + 1, j + 1] * dst_y[i + 1, j + 1]) ** 0.5
    dst = cv2.convertScaleAbs(dst)
    return dst


def laplacian_4(src):
    """
    四邻域 [[0,-1,0],[-1,4,-1],[0,-1,0]]
    :param src:
    :return:
    """
    dst = np.zeros_like(src, dtype=np.float32)
    op = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    h, w = src.shape[:2]
    for i in range(h - 2):
        for j in range(w - 2):
            dst[i + 1, j + 1] = abs(np.sum(src[i: i + 3, j: j + 3] * op))
    dst = cv2.convertScaleAbs(dst)
    return dst


def laplacian_8(src):
    """
    八邻域 [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    :param src:
    :return:
    """
    dst = np.zeros_like(src, dtype=np.float32)
    op = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    h, w = src.shape[:2]
    for i in range(h - 2):
        for j in range(w - 2):
            dst[i + 1, j + 1] = abs(np.sum(src[i: i + 3, j: j + 3] * op))
    dst = cv2.convertScaleAbs(dst)
    return dst


if __name__ == '__main__':
    # 读取图片
    img_gray = cv2.imread('src/emilia.jpg', cv2.IMREAD_GRAYSCALE)

    # 处理函数
    img_robert = robert(img_gray)
    img_sobel = sobel(img_gray)
    img_laplacian4 = laplacian_4(img_gray)
    img_laplacian8 = laplacian_8(img_gray)

    # 结果汇总
    img_show_list = [img_gray, img_robert, img_sobel, img_laplacian4, img_laplacian8]
    title_list = ['raw', 'robert', 'sobel', 'laplacian-4', 'laplacian-8']
    # 开始绘图
    nrow, ncol = 2, 3
    plt.figure()
    for i, img in enumerate(img_show_list):
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(img, 'gray')
        plt.title(title_list[i])
    plt.show()