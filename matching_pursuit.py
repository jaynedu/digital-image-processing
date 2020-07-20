# -*- coding: utf-8 -*-
# @Date    : 2020/7/20 6:52 下午
# @Author  : Du Jing
# @FileName: matching_pursuit
# ---- Description ----
#

import os, sys
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


class MatchingPursuit(object):
    def __init__(self, M, N):
        assert N >= M, 'N < M'
        self.max_iter = 10000                       # 迭代次数
        self.m = M                                  # 词典空间的维度
        self.n = N                                  # 词典D中向量个数
        self.D = self._dictionary_constructor()     # 词典库
        self.Y = self._input_constructor()          # 输入[M, ]
        self.A = np.zeros((N, 1))                   # 待求
        self.matching_pursuit()

    def _dictionary_constructor(self):
        '''
        产生词典库[M, N]
        '''
        H = np.random.random((self.m, self.n))
        I = np.identity(self.m)
        D = np.matmul(I, H)
        for j in range(self.n):
            D[:, j] = D[:, j] / np.linalg.norm(D[:, j])
        return D

    def _input_constructor(self):
        '''
        产生输入
        '''
        x = np.random.uniform(1, 10, (self.m, ))
        # x = np.linspace(0, 2 * np.pi, self.m)
        # x = np.sin(x)
        x *= 1 / np.linalg.norm(x)
        return x

    def matching_pursuit(self):
        '''
        theory:     self.D x self.A = self.Y
        known:      self.D, self.Y
        objective:  self.A
        '''
        D = self.D
        D_T = np.transpose(self.D)
        index = np.zeros((self.n, ))
        coefs = np.zeros((self.n, ))
        r = self.Y

        for epoch in range(self.max_iter):
            proj = np.dot(D_T, r)
            position = np.argmax(np.abs(proj))
            word = D[:, position]
            index[position] += 1
            coefs[position] += proj[position]
            r = r - np.dot(word, r) * word
            l2 = np.linalg.norm(r)
            if np.isclose(l2, 0):
                # print('N = %d - break - iterations = %d' % (self.n, epoch))
                break
        print('N = %d - break - iterations = %d' % (self.n, epoch))
        # 将系数存入系数矩阵A中
        self.A = coefs

    def visualize(self):
        print('origin: ', self.Y)
        output = np.dot(self.D, self.A)
        print('pursuit: ', output)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.Y)
        plt.plot(output)
        plt.title('N = %s' % self.n)
        plt.subplot(2, 1, 2)
        plt.plot(self.A)
        plt.savefig('n=%s.png' % self.n)
        plt.show()


if __name__ == '__main__':
    M = 30
    for N in [40, 100, 500, 1000, 10000]:
        mp = MatchingPursuit(M, N)
        # mp.visualize()

