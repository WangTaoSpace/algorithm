'''
ID3  信息增益
C4.5 信息增益/特征的信息熵
CART分类树 利用基尼指数做最优特征选择和最优切分点选择,终止条件

CART回归树（优化切分成两部分的平方误差和来寻找最佳切分点，所以也叫最小二乘回归树）

'''
#encoding=utf-8
import cv2
import time
import logging
import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
total_class = 10

# 二值化
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv_img = cv2.adaptiveThreshold(cv_img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 7, 10)
    return cv_img
#计算最优特征和最优切分点,进行数据集分割
def calcGini(features,label):
    Ginis = []
    for i in range(features.shape[1]):
        GiniA = calcGiniA(features[:, i], label)
        Ginis.append((i, GiniA[0], GiniA[1]))
        print(i, GiniA)
    Ginis = min(Ginis, key=lambda x: x[2]) # 列index 最佳切分点 基尼值
    # 根据最佳切分点进行数据分割
    featuresD1 = features[features[:, Ginis[0]] == Ginis[1]]
    featuresD2 = features[features[:, Ginis[0]] != Ginis[1]]
    # 删除用过的特征值
    featuresD1 = np.delete(featuresD1, Ginis[0], axis=1)
    featuresD2 = np.delete(featuresD2, Ginis[0], axis=1)
    return featuresD1, featuresD2

# 计算传入特征A的最优切分点
def calcGiniA(featureA,label):
    featureA = featureA.astype(np.float64)
    label = label.astype(np.int)
    #获得特征A中可能的取值a
    a = np.unique(featureA)
    featureA_label = np.vstack((featureA,label)).transpose((1,0))
    GiniA = []
    for a_cell in a:
        D1 = featureA_label[featureA_label[:,0] == a_cell]
        D2 = featureA_label[featureA_label[:,0] != a_cell]
        _, D1 = np.unique(D1[:,1],return_counts=True)
        _, D2 = np.unique(D2[:,1],return_counts=True)
        D1_sums = np.sum(D1)
        D2_sums = np.sum(D2)
        D1 = 1- np.sum(np.power(D1/D1_sums,2))
        D2 = 1- np.sum(np.power(D2/D2_sums,2))
        GiniA_a = (D1_sums*D1)/(D1_sums+ D2_sums) + (D2_sums*D2)/(D1_sums+ D2_sums)
        GiniA.append((a_cell,GiniA_a))
    return min(GiniA, key=lambda x: x[1])  # 根据基尼指数输出最优切分点

if __name__ == '__main__':
    print('Start read data')
    raw_data = pd.read_csv('../data/train.csv', header=0)
    data = raw_data.values
    imgs = data[:, 1:]
    labels = data[:, 0]
    features = imgs
    # features = binaryzation(imgs)
    # 选取 4/5 数据作为训练集， 1/5 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2)
    # 开始计算最优特征和切分点
    featureD1, featureD2 = calcGini(train_features, train_labels)


