# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:03:32 2017

@author: Cindy
"""

#引入函式庫畫圖
import matplotlib.pyplot as plt
import numpy as np

accuracy = [93.658536585365866, 95.121951219512198, 92.682926829268297, 94.634146341463406, 94.146341463414629, 95.609756097560975, 95.121951219512198, 96.097560975609753, 95.121951219512198, 95.609756097560975, 94.780487804878035]
top5_accuracy = [92.682926829268297, 95.121951219512198, 92.682926829268297, 94.634146341463406, 94.634146341463406, 96.58536585365853, 95.121951219512198, 96.097560975609753, 94.634146341463406, 96.097560975609753, 94.829268292682926]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
times = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'avg']
plt.title('Logistic Regression experimental results')


plt.plot(x, accuracy, label='accuracy')
plt.plot(x, top5_accuracy, label='top5_accuracy')
plt.legend(loc='lower left')
plt.xticks(x, times)
plt.yticks(np.arange(90,101,1))
plt.xlabel('times')
plt.ylabel('percentage(%)')
plt.savefig('./Logistic Regression experimental results.jpg', dpi=300)