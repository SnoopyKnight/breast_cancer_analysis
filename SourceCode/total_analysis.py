# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:03:32 2017

@author: Cindy
"""

#引入函式庫畫圖
import matplotlib.pyplot as plt
import numpy as np

Perceptron = [99.093567251461977, 95.116959064327489]
SVM = [99.824561403508767, 95.614035087719301]
Naive_Bayes = [98.391812865497059, 94.502923976608187]
Logistic_Regression = [94.780487804878035, 94.829268292682926]


kind = ['accuracy', 'top5_accuracy']
plt.title('experimental results comparison')

y_pos = np.arange(len(kind))
plt.bar(y_pos, Perceptron, 0.2, alpha=1, label='Perceptron')
plt.bar(y_pos+0.2, SVM, 0.2, alpha=1, label='SVM')
plt.bar(y_pos+0.4, Naive_Bayes, 0.2, alpha=1, label='Naive Bayes')
plt.bar(y_pos+0.6, Logistic_Regression, 0.2, alpha=1, label='Logistic Regression')
plt.xticks(y_pos+0.2, kind)
plt.ylabel('percentage(%)')
plt.legend(loc='lower right')
plt.savefig('./experimental results comparison', dpi=300)