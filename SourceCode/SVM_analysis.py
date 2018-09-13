# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:03:32 2017

@author: Cindy
"""

#引入函式庫畫圖
import matplotlib.pyplot as plt
import numpy as np

accuracy = [100.0, 100.0, 99.415204678362571, 100.0, 99.707602339181292, 100.0, 100.0, 100.0, 99.122807017543863, 100.0, 99.824561403508767]
top5_accuracy = [95.029239766081872, 96.198830409356731, 95.029239766081872, 95.32163742690058, 94.444444444444443, 96.783625730994146, 96.491228070175438, 97.076023391812853, 93.859649122807014, 95.906432748538009, 95.614035087719301]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
times = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'avg']
plt.title('SVM experimental results')


plt.plot(x, accuracy, label='accuracy')
plt.plot(x, top5_accuracy, label='top5_accuracy')
plt.legend(loc='lower left')
plt.xticks(x, times)
plt.yticks(np.arange(90,101,1))
plt.xlabel('times')
plt.ylabel('percentage(%)')
plt.savefig('./SVM experimental results.jpg', dpi=300)