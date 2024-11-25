# -*- coding: UTF-8 -*-

import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# 定義模型
with pm.Model() as model:
    # 定義先驗分佈
    p = pm.Beta('p', alpha=2, beta=2)
    # 定義觀測資料
    observations = pm.Bernoulli('obs', p=p, observed=[1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    # 使用NUTS演算法進行採樣
    trace = pm.sample(1000, tune=1000, return_inferencedata=True)
# 視覺化結果
az.plot_trace(trace)
az.plot_posterior(trace)
plt.show()