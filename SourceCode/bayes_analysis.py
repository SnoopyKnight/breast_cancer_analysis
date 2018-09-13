# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:03:32 2017

@author: Cindy
"""

#引入函式庫畫圖
import matplotlib.pyplot as plt
import numpy as np

accuracy = [97.953216374269005, 98.830409356725141, 97.660818713450297, 98.538011695906434, 97.076023391812853, 98.538011695906434, 98.830409356725141, 98.538011695906434, 98.245614035087712, 99.707602339181292, 98.391812865497059]
top5_accuracy = [94.152046783625735, 94.73684210526315, 93.567251461988292, 94.152046783625735, 94.73684210526315, 95.32163742690058, 94.444444444444443, 95.32163742690058, 93.274853801169584, 95.32163742690058, 94.502923976608187]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
times = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'avg']
plt.title('Naive Bayes experimental results')


plt.plot(x, accuracy, label='accuracy')
plt.plot(x, top5_accuracy, label='top5_accuracy')
plt.legend(loc='under left')
plt.xticks(x, times)
plt.yticks(np.arange(90,101,1))
plt.xlabel('times')
plt.ylabel('percentage(%)')
plt.savefig('./Naive Bayes experimental results.jpg', dpi=300)