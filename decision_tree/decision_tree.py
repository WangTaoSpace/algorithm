'''
ID3  信息增益
C4.5 信息增益/特征数  信息增益比
CART分类树 利用基尼指数做最优特征选择和最优切分点选择,终止条件：样本数少于预定阈值 样本集的基尼指数小于预定阈值 没有更多特征

总结，根据节点的分裂规则可以得出，ID3 C4.5属于极大似然估计概率模型，适合少特征，小样本。CART更加适合大样本

CART回归树（优化切分成两部分的平方误差和来寻找最佳切分点，所以也叫最小二乘回归树）

code infer：https://blog.csdn.net/slx_share/article/details/79992846
'''

#encoding=utf-8
import cv2
import time
import logging
import numpy as np
import pandas as pd
import math
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
total_class = 10

class node:
    def __init__(self,feature = -1,val=None, res=None, right=None,left=None):
        self.feature = feature # 特征
        self.val = val # 特征对应的值
        self.res = res # 叶节点标记
        self.right = right
        self.left = left


class CART_CL:
    def __init__(self,epsilon= 1e-3, min_sample=1):
        self.epsilon = epsilon
        self.min_sample = min_sample # 叶节点含有的最少样本数
        self.gini = 1  # 上一个基尼值，为了求解基尼值的下降
        self.tree = None

    #计算最优特征和最优切分点,进行数据集分割
    def calcGini(self, features, label):
        Ginis = []
        for i in range(features.shape[1]):
            GiniA = self.calcGiniA(features[:, i], label)
            Ginis.append((i, GiniA[0], GiniA[1]))
        Ginis = min(Ginis, key=lambda x: x[2]) # 列index 最佳切分点 基尼值
        # 根据最佳切分点进行数据分割
        features_label = np.hstack((features,label.reshape((-1,1))))

        D1 = features_label[features_label[:, Ginis[0]] == Ginis[1]]
        D2 = features_label[features_label[:, Ginis[0]] != Ginis[1]]
        print(D1.shape,D2.shape)
        features_D1, label_D1 = D1[:, :-1], D1[:, -1]
        features_D2, label_D2 = D2[:, :-1], D2[:, -1]
        res1 = Counter(label_D1).most_common(1)[0][0]
        res2 = Counter(label_D2).most_common(1)[0][0]

        return (features_D1, label_D1, res1), (features_D2, label_D2, res2), Ginis

    # 计算传入特征A的最优切分点
    def calcGiniA(self, featureA, label):
        featureA = featureA.astype(np.float64)
        label = label.astype(np.int)
        #获得特征A中可能的取值a
        a = np.unique(featureA)
        featureA_label = np.vstack((featureA,label)).transpose((1,0))
        GiniA = []
        for a_cell in a:
            D1 = featureA_label[featureA_label[:,0] == a_cell]
            D2 = featureA_label[featureA_label[:,0] != a_cell]
            if D1.shape[0] < 1 or D2.shape[0] < 1:  #消除切分后某一方为空的情况
                continue
            _, D1 = np.unique(D1[:,1],return_counts=True)
            _, D2 = np.unique(D2[:,1],return_counts=True)
            D1_sums = np.sum(D1)
            D2_sums = np.sum(D2)
            D1 = 1- np.sum(np.power(D1/D1_sums,2))
            D2 = 1- np.sum(np.power(D2/D2_sums,2))
            GiniA_a = (D1_sums*D1)/(D1_sums+ D2_sums) + (D2_sums*D2)/(D1_sums+ D2_sums)
            GiniA.append((a_cell,GiniA_a))
        try:
            return min(GiniA, key=lambda x: x[1])  # 根据基尼指数输出最优切分点
        except:
            return (1, 1)  # 该特征不能进行切分，输出特殊值

    def buildTree(self, features, labels, gini = 1):
        if labels.shape[0] <= self.min_sample: # 数据集小于阈值直接设置为叶节点
            return node(res=Counter(labels).most_common(1)[0][0])
        (features_D1, label_D1, res1), (features_D2, label_D2, res2), Ginis = self.calcGini(features, labels)
        if gini - Ginis[2] <= self.epsilon:  # 如果基尼指数的下降值小于阈值，直接返回成叶节点
            return node(res=Counter(labels).most_common(1)[0][0])
        else:
            left = self.buildTree(features_D1,label_D1, Ginis[2])
            right = self.buildTree(features_D2,label_D2, Ginis[2])
            return node(feature=Ginis[0],val=Ginis[1],right = right, left = left)
    def fit(self,features,labels):
        self.tree = self.buildTree(features,labels)

    def predict(self,features):
        result = []
        def helper(x, tree):
            if tree.res is not None: # 表明到达了叶子节点
                return tree.res
            else:
                if x[tree.feature] == tree.val:
                    branch = tree.left
                else:
                    branch = tree.right
                return helper(x,branch)
        for cell in features:
            result.append(helper(cell,self.tree))
        return np.array(result)


    def disp_tree(self):
        # 打印树
        self.disp_helper(self.tree)
        return
    def disp_helper(self, current_node):
        # 前序遍历
        print(current_node.feature, current_node.val, current_node.res)
        if current_node.res is not None:
            return
        self.disp_helper(current_node.left)
        self.disp_helper(current_node.right)
        return

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
        features, labels, test_size=0.3)
    print(train_features.shape,train_labels.shape)
    print("start building tree")
    cart = CART_CL(min_sample=2)
    cart.fit(train_features,train_labels)
    print("end building tree")
    print("print tree")
    print(cart.disp_tree())
    print("start predict")
    test_predict = cart.predict(test_features)
    score = accuracy_score(test_labels,test_predict)
    print("score:",score)
