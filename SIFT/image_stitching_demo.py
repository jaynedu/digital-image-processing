"""
190785 李克
"""
import numpy as np
import cv2
import mySIFT


def stitch(imgs, ratio=0.75):
    img2, img1 = imgs
    # 获取特征点和描述符
    kps1, des1 = mySIFT.extract_feature(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    kps2, des2 = mySIFT.extract_feature(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    kp1 = np.float32([kp.pt for kp in kps1])
    kp2 = np.float32([kp.pt for kp in kps2])

    # 匹配特征点并计算仿射变换矩阵
    R = matchKeyPoints(kp1, kp2, des1, des2, ratio)

    # 如果没有足够的最佳匹配点，R为None
    if R is None:
        return None
    good, M, mask = R

    # 对img1透视变换，M是ROI区域矩阵， 变换后的大小是(img1.w+img2.w, img1.h)
    result = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    cv2.imshow('1', result)
    cv2.waitKey()
    # 将img2的值赋给结果图像
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    return result


def matchKeyPoints(kp1, kp2, des1, des2, ratio):
    matcher = cv2.DescriptorMatcher_create('BruteForce')
    matches = matcher.knnMatch(des1, des2, 2)

    # 获取理想匹配
    good = []
    for m in matches:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            good.append((m[0].trainIdx, m[0].queryIdx))

    # 最少要有四个点才能做透视变换
    if len(good) > 4:
        # 获取特征点的坐标
        src_pts = np.float32([kp1[i] for (_, i) in good])
        dst_pts = np.float32([kp2[i] for (i, _) in good])

        # 通过两个图像的特征点计算变换矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=4.0)

        # 返回最佳匹配点、变换矩阵和掩模
        return good, M, mask
    # 如果不满足最少四个 就返回None
    return None


if __name__ == '__main__':
    imgA = cv2.imread('Flower_A.jpg')
    imgB = cv2.imread('Flower_B.jpg')
    stitched = stitch([imgA, imgB])
    cv2.imwrite('Stitching.jpg', stitched)

