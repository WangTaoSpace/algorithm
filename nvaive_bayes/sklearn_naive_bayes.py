'''
朴素贝叶斯分类规则
1.强假设，特征独立
2.分类规则 y = arg max P(y) * ∏ (P(xi/y))
sklearn中提供的方法
1. Gaussian Naive Bayes
2. Multinomial
3. Complement
4. Bernoulli
5. Categorical
'''
import pandas as pd
import numpy as np
import cv2
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB,CategoricalNB
# 利用opencv获取图像hog特征
def get_hog_features(trainset):
    features = []
    hog = cv2.HOGDescriptor('../hog.xml')
    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)
        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)
    features = np.array(features)
    features = np.reshape(features,(-1,324))
    return features

if __name__ == '__main__':
    print('Start read data')
    time_1 = time.time()
    raw_data = pd.read_csv('../data/train.csv',header=0)
    data = raw_data.values
    imgs = data[:,1:]
    labels = data[:,0]
    #features = get_hog_features(imgs)  # hog特征维度是324  0.5580952380952381
    features = imgs  #0.5560714285714285
    # 选取 4/5 数据作为训练集， 1/5 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.2)

    clf = GaussianNB() # 0.5560714285714285
    #clf = MultinomialNB() # 0.8230952380952381
    #clf = ComplementNB() #  0.7144047619047619
    #clf = BernoulliNB() #  0.8323809523809523
    #clf = CategoricalNB()

    print("start train")
    clf.fit(train_features,train_labels)
    print("start test")
    test_predict = clf.predict(test_features)
    score = accuracy_score(test_labels,test_predict)
    print ("The accruacy socre is ", score)
