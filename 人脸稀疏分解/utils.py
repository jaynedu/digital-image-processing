from scipy import linalg
import numpy as np
import numbers

__all__ = ['extract_patches_2d', 'randomized_svd']


def extract_patches_2d(img, patch_size, max_patches, step):
    """提取二维图像的patch
    :param img: 输入图像
    :param patch_size: patch尺寸
    :param max_patches: patch最大提取数量
    :param step: 滑动步长
    :return: 完成提取的patch, shape = (n_patches, patch_height, patch_width)
    """
    i_h, i_w = img.shape[:2]
    p_h, p_w = patch_size

    # patch的宽和高不能超过图像的宽和高
    assert i_h >= p_h
    assert i_w >= p_w
    # 输入的max_patches必须是整数
    assert isinstance(max_patches, numbers.Integral)

    image = np.reshape(img, (i_h, i_w, -1))  # 保证输入图像是三维
    i_h, i_w, n_channels = image.shape
    n_h = i_h - p_h + 1  # height中可以提取patch的数量
    n_w = i_w - p_w + 1  # width中可以提取patch的数量
    extracted_patches = np.zeros((n_h, n_w, 1, p_h, p_w, n_channels))  # 构造待提取的patch

    # 提取patch
    for y in range(0, n_h, step):
        for x in range(0, n_w, step):
            patch = image[y:y + p_h, x:x + p_w]
            extracted_patches[y, x, 0] = patch

    # 计算patch的数量
    all_patches = n_h * n_w
    if max_patches:
        n_patches = max_patches if max_patches < all_patches else all_patches
        i_s = np.random.randint(n_h, size=n_patches)
        j_s = np.random.randint(n_w, size=n_patches)
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches

    patches = np.reshape(patches, (-1, p_h, p_w, n_channels))
    return patches


def randomized_range_finder(A, size, random_state):
    Q = random_state.normal(size=(A.shape[1], size))  # 构建高斯随机矩阵，这样A*Q就相当于是A的range，得到的行数82500远大于列数100，可以认为其列数是不相关的
    Q = Q.astype(A.dtype, copy=False)
    for i in range(4):  # 反复相乘，得到更具完备性的A的range（不可能完全一致，但最终Q的range与A是近似的），做LU分解是为了保证数据稳定性，类似于归一化
        Q, _ = linalg.lu(np.dot(A, Q), permute_l=True)
        Q, _ = linalg.lu(np.dot(A.T, Q), permute_l=True)
    Q, _ = linalg.qr(np.dot(A, Q), mode='economic')  # 最终结果进行QR分解，能够得到列正交阵（行数很大）
    return Q


def svd_flip(u, v):
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    v *= signs[:, np.newaxis]
    return u, v


def randomized_svd(M, n_components, random_state):
    """
    一种快速svd方法，基本思想是构建一个B=QT*A，使B的列数远小于A的列数但B和A的列向量span得到的子空间是相似的,或者说值域R(A)与R(B)相似
    在本实例中，A.shape为82500，100，B.shape为100，100
    :param M: 训练数据
    :param n_components: 词典规模
    :param random_state: 随机数发生器
    :return: 初始化的词典
    """
    n_random = n_components + 10  # 因为我们做的是近似计算，因此有必要在计算中略微扩展所需要的维数（词典数）来弥补准确性的损失
    Q = randomized_range_finder(M, n_random, random_state)  # 构建矩阵Q
    B = np.dot(Q.T, M)
    Uhat, s, V = linalg.svd(B, full_matrices=False)  # 对B做SVD，这个计算量就远小于对A直接做SVD
    U = np.dot(Q, Uhat)  # B=USV,A≈QQTA（约等于是因为不是严格正交阵，只有列正交），B=QTA可推得A≈（QU）SV，因为 U 和 Q 都是列orthonormal的，QU也一定列orthonormal
    U, V = svd_flip(U, V)  # 对结果符号的修改（将U中绝对值最大的设为正），V随之改变
    return V[:n_components, :]
