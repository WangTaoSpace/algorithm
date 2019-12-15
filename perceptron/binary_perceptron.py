import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score

class Perceptron(object):

    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000  # 这个参数太大，会出现过拟合效果反而不好

    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        return int(wx > 0)

    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) +1)  # w = 785  truth 784 ，多加了一位为b


        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1)  # 随机选取数据
            x = list(features[index])
            x.append(1.0)  # 多加了一位，因为是单独加一个b，所以这里填个1，就1*b = b
            y = 2 * labels[index] - 1  # 规范成1，-1
            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])  # d=1/|W| * (WX+b) 这里没有除以W的L2范式

            if wx * y > 0:    # 表示对这个数据正确分类，对应于误分类的公式
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue  # 如果分类正确的话，不对参数进行修改

            for i in range(len(self.w)):
                self.w[i] += self.learning_step * (y * x[i])  # 这里对w和b都进行了梯度更新

    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels


if __name__ == '__main__':

    print('Start read data')

    time_1 = time.time()
    raw_data = pd.read_csv('../data/train_binary.csv', header=0)  # header = 0 忽略抬头
    data = raw_data.values
    print("data",type(data))
    imgs = data[:, 1:]
    labels = data[:, 0]

    # 选取 4/5 数据作为训练集， 1/5 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.2)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')

    print('Start training')
    p = Perceptron()
    p.train(train_features, train_labels)

    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' second', '\n')

    print ('Start predicting')
    test_predict = p.predict(test_features)
    time_4 = time.time()
    print ('predicting cost ', time_4 - time_3, ' second', '\n')

    score = accuracy_score(test_labels, test_predict)
    print ("The accruacy socre is ", score)