"""
infer https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/85333677

nearest neighbor algorithms
1. brute force
2. KDtree
3. ball tree
依次对维度和数据量的增加有效
在kdtree和ball tree 设置了参数leaf_size(非节点，叶子的数量),来优化搜索，因为在小数据量上是brute force占优势
"""
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import cv2
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    features = get_hog_features(imgs)
    # 选取 4/5 数据作为训练集， 1/5 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.2)
    knn = KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree',leaf_size=30,p=2,metric='minkowski',metric_params=None,n_jobs=-1)
    print("start train")
    knn.fit(train_features,train_labels)
    print("start test")
    test_predict = knn.predict(test_features)
    score = accuracy_score(test_labels,test_predict)
    print ("The accruacy socre is ", score)

