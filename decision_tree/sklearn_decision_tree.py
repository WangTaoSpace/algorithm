import pandas as pd
import numpy as np
import cv2
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
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
def binaryzation(imgs):
    features = []
    for img in  imgs:
        cv_img = img.astype(np.uint8)
        cv_img = cv2.adaptiveThreshold(cv_img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 7, 10)
        features.append(cv_img)
    features = np.array(features)
    features = np.reshape(features,(-1,784))
    return features
if __name__ == '__main__':
    print('Start read data')
    time_1 = time.time()
    raw_data = pd.read_csv('../data/train.csv',header=0)
    data = raw_data.values
    imgs = data[:,1:]
    labels = data[:,0]
    features = get_hog_features(imgs)  #这里hog特征表现要好很多
    #features = imgs
    #features = binaryzation(imgs)
    # 选取 4/5 数据作为训练集， 1/5 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2)

    clf = DecisionTreeClassifier(max_leaf_nodes=1000, random_state=0) # max_depth和max_leaf_nodes 防止过拟合

    print("start train")
    clf.fit(train_features,train_labels)
    print("start test")
    test_predict = clf.predict(test_features)
    score = accuracy_score(test_labels,test_predict)
    print ("The accruacy socre is ", score)
