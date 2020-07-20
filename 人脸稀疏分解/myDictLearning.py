from .mySparsencode import omp, mp, bp
from .utils import randomized_svd
from scipy import linalg
import numpy as np
import itertools


class DictLearning(object):
    def __init__(self,
                 n_components=None,
                 n_iter=1000,
                 batch_size=3,
                 transform_algorithm='omp',
                 ):
        self.n_components = n_components
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.transform_algorithm = transform_algorithm
        self.random_state = np.random.mtrand._rand
        self.dictionary = None

    def dict_learning(self, X):
        n_components = self.n_components if self.n_components else X.shape[1]
        n_samples, n_features = X.shape
        dictionary = randomized_svd(X, n_components, self.random_state)  # 词典初始化的一般采用样本右奇异矩阵保证正交性，这里使用了一种快速SVD方法
        r = len(dictionary)
        dictionary = np.r_[dictionary, np.zeros((n_components - r, dictionary.shape[1]))]  # 0填充
        dictionary = dictionary.T

        # 打乱训练数据并循环
        X_train = X.copy()
        self.random_state.shuffle(X_train)
        batches = self._gen_batches(n_samples, self.batch_size)  # 根据BATCH_SIZE生成数据batch
        batches = itertools.cycle(batches)

        A = np.zeros((n_components, n_components))
        B = np.zeros((n_features, n_components))

        for ii, batch in zip(range(0, self.n_iter), batches):
            this_X = X_train[batch]
            # 计算当前的稀疏编码
            if self.transform_algorithm == 'omp':
                this_code = omp(dictionary, this_X.T, 20)
            elif self.transform_algorithm == 'mp':
                this_code = mp(dictionary, this_X.T, 20)
            elif self.transform_algorithm == 'bp':
                this_code = bp(dictionary, this_X.T, 20)
            else:
                raise ValueError('Invalid Transform Algorithm!')

            if ii < self.batch_size - 1:  # 对batch进行批处理，为批量梯度下降法的迭代更新相关参数
                theta = float((ii + 1) * self.batch_size)
            else:
                theta = float(self.batch_size ** 2 + ii + 1 - self.batch_size)
            beta = (theta + 1 - self.batch_size) / (theta + 1)
            A *= beta
            A += np.dot(this_code, this_code.T)
            B *= beta
            B += np.dot(this_X.T, this_code.T)
            # 所以这里输入的是根据batch_size加权后的编码code和样本X，并且保留了evolution的history
            dictionary = self._update_dict(dictionary, B, A, random_state=self.random_state)
        self.dictionary = dictionary.T
        return self

    @staticmethod
    def _gen_batches(n, batch_size):
        start = 0
        for _ in range(int(n // batch_size)):
            end = start + batch_size
            yield slice(start, end)
            start = end
        if start < n:
            yield slice(start, n)

    @staticmethod
    def _update_dict(dictionary, Y, code, random_state):
        n_components = len(code)
        n_features = Y.shape[0]
        # 用blas包加速运算
        gemm, = linalg.get_blas_funcs(('gemm',), (dictionary, code, Y))  # 矩阵乘法
        ger, = linalg.get_blas_funcs(('ger',), (dictionary, code))  # 向量乘法
        nrm2, = linalg.get_blas_funcs(('nrm2',), (dictionary,))  # 求二范
        R = gemm(-1.0, dictionary, code, 1.0, Y)  # 整体的误差
        for k in range(n_components):  # 按列更新词典
            R = ger(1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)  # 除去第k列的误差R，等于样本Y减去字典其余列和编码的乘积
            dictionary[:, k] = np.dot(R, code[k, :])  # 求使R-dictionary[:, k]*code[k, :]最小的dictionary[:, k],使用最小二乘法求解
            atom_norm = nrm2(dictionary[:, k])
            if atom_norm < 1e-10:  # 由于code是稀疏矩阵，如果计算得到的dictionary[:, k]很小说明该列编码应该为0，可以随机设置
                dictionary[:, k] = random_state.randn(n_features)
                atom_norm = nrm2(dictionary[:, k])
                dictionary[:, k] /= atom_norm  # 归一化
            else:
                dictionary[:, k] /= atom_norm
                # 重新计算整体误差
                R = ger(-1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)
        return dictionary
