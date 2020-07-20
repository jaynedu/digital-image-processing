from myDictLearning import DictLearning
from mySparsencode import omp, mp, bp
from utils import extract_patches_2d
import matplotlib.pyplot as plt
import numpy as np
import imageio
import time
import cv2
import os


'''全局变量'''
FACE_DETECTOR = cv2.CascadeClassifier('haar_alt.xml')  # 人脸检测器
DATABASE_DIR = './yalefaces'  # 人脸数据集目录
DATABASE_SIZE = 154  # 训练集大小
'''训练参数'''
NUM_PATCHES = 841  # Patch数量
FACE_SIZE = 64  # 归一化人脸图像大小
PATCH_SIZE = 8  # Patch尺寸
DICT_SIZE = 256  # 字典大小
BATCH_SIZE = 16  # 训练批次大小


def prepare_train_data():
    """准备训练数据"""
    data = np.zeros((DATABASE_SIZE * NUM_PATCHES, PATCH_SIZE ** 2))
    # 读取数据集下所有图像
    for i, train_img in enumerate(os.listdir(os.path.join(DATABASE_DIR, 'train'))):
        img = np.array(imageio.imread(os.path.join(DATABASE_DIR, 'train', train_img)))
        # 检测图像中的人脸
        bboxes = FACE_DETECTOR.detectMultiScale(image=img, scaleFactor=1.1, minNeighbors=3)
        if len(bboxes) != 1:
            raise ValueError(train_img, 'Face Detection Error!')

        # 裁剪出人脸图像并做尺寸归一化
        bbox = bboxes[0]
        top = bbox[1]
        right = bbox[0] + bbox[3]
        bottom = bbox[1] + bbox[2]
        left = bbox[0]
        face = img[top:bottom, left:right]
        face = cv2.resize(face, (FACE_SIZE, FACE_SIZE)) / 255.0

        # 提取出若干Patch
        patches = extract_patches_2d(face, (PATCH_SIZE, PATCH_SIZE), NUM_PATCHES, step=2)
        data[i * NUM_PATCHES: (i + 1) * NUM_PATCHES, :] = patches.reshape(len(patches), -1)

    # 训练数据归一化
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    return data


def sparse_reconstruction(img_path, dictionary, transform_algorithm, threshold):
    """对图像进行稀疏重建"""
    # 读取图像并检测人脸
    img = np.array(imageio.imread(img_path))
    bboxes = FACE_DETECTOR.detectMultiScale(image=img, scaleFactor=1.1, minNeighbors=3)
    if len(bboxes) != 1:
        raise ValueError('Face Detection Error!')

    # 裁剪出人脸图像并做尺寸归一化
    bbox = bboxes[0]
    top = bbox[1]
    right = bbox[0] + bbox[3]
    bottom = bbox[1] + bbox[2]
    left = bbox[0]
    face = img[top:bottom, left:right]
    face = cv2.resize(face, (FACE_SIZE, FACE_SIZE)) / 255.0
    mean = np.mean(face)
    std = np.std(face)
    face_origin = face.copy()
    face -= mean
    face /= std

    # 图片分块
    patches = []
    for i in range(FACE_SIZE // PATCH_SIZE):
        for j in range(FACE_SIZE // PATCH_SIZE):
            patch = face[i * PATCH_SIZE:(i + 1) * PATCH_SIZE, j * PATCH_SIZE:(j + 1) * PATCH_SIZE]
            patches.append(patch.reshape(-1))
    patches = np.stack(patches)

    # 逐块进行稀疏编码
    if transform_algorithm == 'omp':
        coefficients = omp(dictionary.T, patches.T, threshold)
    elif transform_algorithm == 'mp':
        coefficients = mp(dictionary.T, patches.T, threshold)
    elif transform_algorithm == 'bp':
        coefficients = bp(dictionary.T, patches.T, threshold)
    else:
        raise ValueError('Invalid Transform Algorithm!')

    # 重建图像
    patches = np.dot(dictionary.T, coefficients).T
    patches = patches.reshape((len(patches), PATCH_SIZE, PATCH_SIZE))
    reconstruction = np.zeros_like(face)
    for i in range(FACE_SIZE // PATCH_SIZE):
        for j in range(FACE_SIZE // PATCH_SIZE):
            reconstruction[i * PATCH_SIZE:(i + 1) * PATCH_SIZE, j * PATCH_SIZE:(j + 1) * PATCH_SIZE] = patches[i * (FACE_SIZE // PATCH_SIZE) + j]

    reconstruction *= std
    reconstruction += mean
    np.clip(reconstruction, 0, 1)
    rmse = np.sqrt(np.mean((face_origin - reconstruction) ** 2))

    # 显示结果
    plt.figure(figsize=(12, 9))
    plt.subplot(221)
    plt.imshow(face, cmap='gray')
    plt.title('Original Image')
    plt.subplot(222)
    plt.imshow(reconstruction, cmap='gray')
    plt.title('Reconstruction(RMSE: {:.4f})'.format(rmse))
    plt.subplot(212)
    plt.bar(np.arange((FACE_SIZE // PATCH_SIZE) ** 2 * DICT_SIZE), coefficients.reshape(-1))
    plt.ylim(-4, 4)
    plt.title('Sparse Representation(Sparsity: {:.2f}%)'.format(len(np.where(coefficients != 0)[0]) * 100 / coefficients.shape[0] / coefficients.shape[1]))
    plt.savefig('./result/' + img_path.split('.')[-2] + '.png')
    # plt.show()
    plt.clf()
    return rmse


if __name__ == '__main__':
    print('正在准备训练数据!')
    train_data = prepare_train_data()
    print('正在训练词典!')
    Model = DictLearning(n_components=DICT_SIZE, batch_size=BATCH_SIZE)

    time_start = time.time()
    Dictionary = Model.dict_learning(train_data).dictionary
    time_end = time.time()
    np.save('dictionary', Dictionary)
    print('词典训练完成，耗时: %ds' % (time_end - time_start))

    Dictionary = np.load('dictionary.npy')

    print('正在进行稀疏编码与图像重建!')
    test_imgs = os.listdir(os.path.join(DATABASE_DIR, 'test'))
    RMSE = 0
    for test_img in test_imgs:
        print('正在处理' + test_img)
        RMSE += sparse_reconstruction(os.path.join(DATABASE_DIR, 'test', test_img), Dictionary, 'omp', 20)
    print('测试集平均RMSE: %.4f' % (RMSE / len(test_imgs)))
