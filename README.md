### 什么是局部加权线性回归（Locally Weighted Linear Regression，LWLR）

​	在原版的线性回归算法中，通过下式构建模型：
$$
\underset{\theta}{arg\ min}\sum_{i=1}^m(y_i-\theta ^Tx_i)^2 \\
output: \theta ^Tx
$$
​	而局部加权线性回归加入高斯核权重，模型为：
$$
\underset{\theta}{arg\ min}\sum_{i=1}^mw_i(y_i-\theta ^Tx_i)^2 \\
output: \theta ^Tx
$$
​	其中，权重$w_i$为：
$$
w_i=exp(-\frac{(x_i-x)^2}{2k^2})
$$
​	

​	存在的缺点：虽然局部线性回归模型增强了模型的泛化能力，但是对应每个预测数据点，都必须使用整个训练数据集重新估计回归方程，计算成本非常高，在工程应用中不现实。

______

### 依赖的环境

- python3.6
  - numpy
  - scikit-learn
  - tqdm

____

### 快速开始

导入模块

```python
from LWLR import LWLR
```

初始化模型

```python
lw = LWLR(0.5)
```

训练模型

```python
lw.fit(X, y)
```

预测

```python
lw.predict(X)
```

