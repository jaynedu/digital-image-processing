# -*- coding: utf-8 -*-
# @Time    : 2020/6/23 17:26
# @Author  : Du Jing
# @FileName: spectrogram.py
# @Usage   : 图像的幅度谱、频谱图

import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_amplitude_angle(src):
    """
    获取图像中的幅度信息和角度信息
    :param src: 原始灰度图
    :return:
    """
    fft = np.fft.fft2(src)  # 计算fft
    fftshift = np.fft.fftshift(fft)  # 把零频点移到频谱中间
    amplitude = np.abs(fftshift)
    angle = np.angle(fftshift)
    return amplitude, angle


def get_show_array(src, use_log=True):
    return np.log(np.abs(src)) if use_log else np.abs(src)


def reconstuct_image(amplitude, angle):
    dst = amplitude * np.cos(angle) + (0 + 1j) * np.sin(angle) * amplitude
    ifftshift = np.fft.ifftshift(dst)
    ifft = np.fft.ifft2(ifftshift)
    return get_show_array(ifft, use_log=False)


def display():
    """
    显示一张图的幅度谱和相位谱
    :return:
    """
    # 读取图片
    img_gray = cv2.imread('src/bowl.tiff', cv2.IMREAD_GRAYSCALE)

    amplitude, angle = get_amplitude_angle(img_gray)
    phase = np.cos(angle) + (0 + 1j) * np.sin(angle)  # 相位谱

    amplitude_show = get_show_array(amplitude) # 幅度谱图像
    phase_show = get_show_array(phase)  # 相位谱图像

    img_show_list = [amplitude_show, phase_show]
    title_list = ['amplitude', 'phase']
    nrow, ncol = 1, 2
    plt.figure()
    for i, img in enumerate(img_show_list):
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(img, 'gray')
        plt.title(title_list[i])
    plt.show()


def exchange():
    """
    交换两幅图像的幅度和相位 重构图像

    => 相位谱包含更多图像信息
    :return:
    """
    img_1 = cv2.imread('src/lena.jpg', cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread('src/bowl.tiff', cv2.IMREAD_GRAYSCALE)

    amp_1, angle_1 = get_amplitude_angle(img_1)
    amp_2, angle_2 = get_amplitude_angle(img_2)

    dst_1 = reconstuct_image(amp_1, angle_2)
    dst_2 = reconstuct_image(amp_2, angle_1)

    img_show_list = [dst_1, dst_2]
    title_list = ['amp1-ang2', 'amp2-ang1']
    nrow, ncol = 1, 2
    plt.figure()
    for i, img in enumerate(img_show_list):
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(img, 'gray')
        plt.title(title_list[i])
    plt.show()


if __name__ == '__main__':
    display()
    exchange()