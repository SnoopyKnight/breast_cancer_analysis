# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:03:32 2017

@author: Cindy
"""

#引入函式庫畫圖
import matplotlib.pyplot as plt
import numpy as np

accuracy = [99.122807017543863, 99.707602339181292, 99.415204678362571, 99.707602339181292, 100.0, 93.274853801169584, 100.0, 99.707602339181292, 100.0, 100.0, 99.093567251461977]
top5_accuracy = [95.029239766081872, 95.906432748538009, 93.859649122807014, 94.444444444444443, 94.73684210526315, 96.491228070175438, 93.859649122807014, 96.783625730994146, 94.73684210526315, 95.32163742690058, 95.116959064327489]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
times = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'avg']
plt.title('Perceptron experimental results')


plt.plot(x, accuracy, label='accuracy')
plt.plot(x, top5_accuracy, label='top5_accuracy')
plt.legend(loc='under left')
plt.xticks(x, times)
plt.yticks(np.arange(90,101,1))
plt.xlabel('times')
plt.ylabel('percentage(%)')
plt.savefig('./Perceptron experimental results.jpg', dpi=300)