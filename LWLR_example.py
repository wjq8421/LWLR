import pandas as pd

from LWLR import LWLR

if __name__ == '__main__':
    train = pd.read_csv("height_train.csv") # 读取训练数据集

    lw = LWLR(0.5)
    lw.fit(train[['father_height', 'mother_height']], train['child_height'])
    result = lw.predict(train[['father_height', 'mother_height']])