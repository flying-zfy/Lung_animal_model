'''
画模型的参数变换曲线图
'''

import pandas as pd
import argparse
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot1030", type=str, default = "/cptjack/totem/zhaofeiyan/DataSet/DataSet8-h5/plot1030.png",help = "path to output loss/accuracy plot")

args = vars(ap.parse_args())
p = '/cptjack/totem/zhaofeiyan/DataSet/DataSet8-h5/2019_10_30_log.csv'
data = pd.read_csv(p)
acc = data['acc']
loss = data['loss']
val_loss = data['val_loss']
val_acc = data['val_acc']

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(loss, color = 'r', linestyle = 'dashed', marker = 'x', label = 'loss')
plt.plot(val_loss, color = 'k', linestyle = 'dashed',marker = 'x', label = 'val_loss')
plt.plot(acc, color = 'r', linestyle = 'dashed', marker = 'o', label = 'acc')
plt.plot(val_acc, color = 'k', linestyle = 'dashed',marker = 'o', label = 'val_acc')
plt.legend(loc = 'best')
ax.set_title('')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss/Accuracy')
plt.savefig(args["plot1030"])
