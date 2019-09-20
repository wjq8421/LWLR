from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import pandas as pd
import numpy as np

class LWLR(object):
    """
    局部加权线性回归（Locally Weighted Linear Regression, LWLR）
    """
    def __init__(self, k=1):
        # 权重系数衰减因子，越小，权重衰减越快
        self.k = k 
    
    def fit(self, X, y):
        # 转化为np.array，方便后续处理
        self.X = X.values 
        self.y = y.values.reshape((-1, 1))
        return self
        
    def predict(self, X):
        result = [] # 保留所有待预测数据的预测值
        X = X.values
        for i in tqdm(range(X.shape[0])):
            # 遍历每个待预测的点
            prediction = self._predict_single(X[i, :])
            result.append(prediction)
        return np.array(result)
        
    def _predict_single(self, example):
        # 算距离，并转化成权重（也是向量）
        dist = np.sum(np.square(self.X - example), axis=1) # shape: (# of samples, )
        # 用LinearRegression预测该点的值
        lr = LinearRegression()
        lr.fit(self.X, self.y, sample_weight=dist)
        example = example.reshape(1, -1) # shape: (1, # of features)
        # 返回预测值
        return lr.predict(example)[0][0]

"""
# 参考示例文档：LWLR_example.py
if __name__ == '__main__':
    train = pd.read_csv("height_train.csv") # 读取训练数据集

    lw = LWLR(0.5)
    lw.fit(train[['father_height', 'mother_height']], train['child_height'])
    result = lw.predict(train[['father_height', 'mother_height']])
"""