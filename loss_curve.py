#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pylab
from pylab import figure, show, legend
from mpl_toolkits.axes_grid1 import host_subplot
 
# read the log file
fp = open('2.txt', 'r')
 
train_iterations = []
train_loss = []
test_iterations = []
#test_accuracy = []
 
 
# 遍历每一行 
for ln in fp:
  # get train_iterations and train_loss
  if '] Iteration ' in ln and 'loss = ' in ln:
    # 以上图第一行为例，以下的命令匹配的是'ion 22000,'
    arr = re.findall(r'ion \b\d+\b,',ln)
    train_iterations.append(int(arr[0].strip(',')[4:]))
    
    # 以下寻找的是0.173712，
    # 训练次数过多时运行下行代码，日志文件会报错“ValueError: invalid literal for float(): 0.069speed: 0.056s / iter”
    # train_loss.append(float(ln.strip().split(' = ')[-1])) 进行以下修改
 
    y = ln.strip().split(' = ')[-1]
    try:
      x = float(y)
    except ValueError:
      ind = y.index('speed')
      x = float(y[0:ind])
    train_loss.append(x)
    
fp.close()
 
host = host_subplot(111)
plt.subplots_adjust(right=0.8) # ajust the right boundary of the plot window
#par1 = host.twinx()
# set labels
host.set_xlabel("iterations")
host.set_ylabel("RPN loss")
#par1.set_ylabel("validation accuracy")
# plot curves
p1, = host.plot(train_iterations, train_loss, label="train RPN loss")
#p2, = par1.plot(test_iterations, test_accuracy, label="validation accuracy")
 
# set location of the legend, 
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
host.legend(loc=1)
 
# set label color
host.axis["left"].label.set_color(p1.get_color())
#par1.axis["right"].label.set_color(p2.get_color())
# set the range of x axis of host and y axis of par1
host.set_xlim([-100,5000])
host.set_ylim([0., 1.6])
plt.draw()
plt.show()
plt.savefig('loss_curve.jpg')