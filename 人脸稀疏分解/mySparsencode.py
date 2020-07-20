from scipy import optimize
import numpy as np


def omp(dictionary, signal, threshold):
    """
    正交匹配追踪算法
    :param signal: 原始图像（mxn）
    :param dictionary: 字典（mxk）
    :param threshold: 稀疏度控制参数
    :return: 稀疏编码系数
    """
    n = signal.shape[1] if len(signal.shape) > 1 else 1  # 原始图像signal中向量的个数，也就是y对应的长度
    k = dictionary.shape[1]  # 字典dictionary中向量的个数
    result = np.zeros((k, n))  # 系数矩阵result中行数等于dictionary中向量的个数，列数等于signal中向量的个数

    for i in range(n):
        indices = []  # 记录选中字典中原子的位置
        coefficients = []  # 存储系数向量
        residue = signal[:, i]
        for j in range(threshold):
            projection = np.dot(dictionary.T, residue)
            # 获取内积向量中元素绝对值的最大值
            max_value = projection.max()
            if abs(projection.min()) >= abs(projection.max()):
                max_value = projection.min()
            pos = np.where(projection == max_value)[0]  # 认为最大值是包含信息最多的
            indices.append(pos.tolist()[0])
            # 使用最小二乘更快的迭代，omp算法的先进之处
            my = np.linalg.pinv(dictionary[:, indices[0: j + 1]])  # 最小二乘,
            coefficients = np.dot(my, signal[:, i])  # 最小二乘
            residue = signal[:, i] - np.dot(dictionary[:, indices[0: j + 1]], coefficients)
            if (residue ** 2).sum() < 1e-10:  # 如果误差很小了，那么即使稀疏度没有达到要求也停止循环
                break
        for t, s in zip(indices, coefficients):
            result[t][i] += s
    return result


def mp(dictionary, signal, threshold):
    """
    匹配追踪算法
    :param signal: 原始图像（mxn）
    :param dictionary: 字典（mxk）
    :param threshold: 稀疏度控制参数
    :return: 稀疏编码系数
    """
    n = signal.shape[1] if len(signal.shape) > 1 else 1  # 原始图像signal中向量的个数，也就是y对应的长度
    k = dictionary.shape[1]  # 字典dictionary中向量的个数
    result = np.zeros((k, n))  # 系数矩阵result中行数等于dictionary中向量的个数，列数等于signal中向量的个数

    for i in range(n):
        indices = []  # 记录选中字典中原子的位置
        coefficients = []  # 存储系数向量
        residue = signal[:, i]
        for j in range(threshold):
            projection = np.dot(dictionary.T, residue)
            # 获取内积向量中元素绝对值的最大值
            max_value = projection.max()
            if abs(projection.min()) >= abs(projection.max()):
                max_value = projection.min()
            pos = np.where(projection == max_value)[0]  # 认为最大值是包含信息最多的
            indices.append(pos.tolist()[0])
            coefficients.append(max_value)
            residue = signal[:, i] - np.dot(dictionary[:, indices[0: j + 1]], np.array(coefficients))
            if (residue ** 2).sum() < 1e-6:  # 如果误差很小了，那么即使稀疏度没有达到要求也停止循环
                break
        for t, s in zip(indices, coefficients):
            result[t][i] += s
    return result


def bp(dictionary, signal, threshold):
    """
    基追踪算法
    :param signal: 原始图像（mxn）
    :param dictionary: 字典（mxk）
    :param threshold: 稀疏度控制参数
    :return : 稀疏编码系数
    """
    n = signal.shape[1] if len(signal.shape) > 1 else 1  # 原始图像signal中向量的个数，也就是y对应的长度
    k = dictionary.shape[1]  # 字典dictionary中向量的个数
    temp = np.zeros((k, n))  # 系数矩阵result中行数等于dictionary中向量的个数，列数等于signal中向量的个数
    result = np.zeros((k, n))

    for i in range(n):
        A_ub = None
        B_ub = None
        c = np.ones([2 * k, 1])  # 优化目标
        A = np.hstack((dictionary, -dictionary))  # A是测量矩阵
        b = signal[:, i]  # 观察信号
        # lb = np.zeros([2*k,1])
        x0 = optimize.linprog(c, A_ub, B_ub, A, b)  # 调用库函数经行线性规划，解决l1范数问题
        # x0对象中的x为稀疏编码系数
        temp[:, i] = x0.x[0:k] - x0.x[k:2 * k]
    idx = np.dstack(np.unravel_index(np.argsort(np.abs(temp.ravel()))[::-1], (k, n)))
    for i in range(int(threshold * n)):
        result[idx[0][i][0], idx[0][i][1]] = temp[idx[0][i][0], idx[0][i][1]]
    return result
