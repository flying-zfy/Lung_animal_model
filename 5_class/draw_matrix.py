
"""
Created on 2019/11/16
@author: zfy
"""

import numpy as np
import glob
import time
import os
from sklearn.metrics import classification_report, confusion_matrix
from scipy.misc import imresize
from keras.models import load_model
from skimage import io
from sklearn.metrics import f1_score
from keras.utils.np_utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import  Image
from sklearn.metrics import accuracy_score

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NasNet_model=load_model('/cptjack/totem/zhaofeiyan/DataSet/5-class/trained-model/Inception_Mobile_1115.h5')

test_folders = ['/cptjack/totem/zhaofeiyan/Data-new/Data1/val/0/',
                '/cptjack/totem/zhaofeiyan/Data-new/Data1/val/1/',
                '/cptjack/totem/zhaofeiyan/Data-new/Data1/val/3/',
                '/cptjack/totem/zhaofeiyan/Data-new/Data1/val/5/',
                '/cptjack/totem/zhaofeiyan/Data-new/Data1/val/7/']

print("\nImages for Testing")
print("=" * 30)

NasNet_model.summary()
test_images = []
test_labels = []
test_img = []
files = []

for index, folder in enumerate(test_folders):
    files = glob.glob(folder + "*.png")
    images = io.imread_collection(files)
    images = [imresize(image, (299,299)) for image in images]    ### Reshape to (299, 299, 3) ###
    labels = [index] * len(images)
    test_images = test_images + images
    test_labels = test_labels + labels
    print("Class: %s. Size: %d" %(folder.split("/")[-2], len(images)))

print("\n")


#def imagenet_processing(image):
#
#    mean = [0.485,0.456,0.406]
#    std = [0.229,0.224,0.225]
#    for i in range(3):
#        image[:,:,i] -= mean[i]
#        image[:,:,i] /= std[i]
#
#    return image

test_images = np.stack(test_images)
test_images = (test_images/255).astype(np.float32)     ### Standardise
#test_images = imagenet_processing(test_images)
test_labels = np.array(test_labels).astype(np.int32)

print(len(test_labels[test_labels==0]))
print(test_labels)

Y_test = to_categorical(test_labels, num_classes = np.unique(test_labels).shape[0])
start_time= time.time()
x_posteriors = NasNet_model.predict(test_images, batch_size = 32)
predictions = np.argmax(x_posteriors, axis=1)
cost_time=time.time()-start_time
acc=accuracy_score(test_labels,predictions)
data=[]

for i in range(len(test_labels)):
    data.append((test_labels[i],x_posteriors[i]))


f1=f1_score(test_labels,predictions,average='weighted')
print("cost_time:",cost_time)
print("acc",acc)
print("F1:",f1)
cr = classification_report(test_labels, predictions, target_names = ["0", "1", "3", "5", "7"], digits = 2)
print(cr, "\n")
print("Confusion Matrix")
print("=" * 30, "\n")

def draw_confusion_matrix_classes():
    cm = confusion_matrix(test_labels, predictions)
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    plt.title('Confusion matrix', size=15)
    plt.colorbar()
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, ["0", "1", "3", "5", "7"], rotation=45, size=10)
    plt.yticks(tick_marks, ["0", "1", "3", "5", "7"], size=10)
    plt.tight_layout()
    plt.ylabel('Actual label', size=15)
    plt.xlabel('Predicted label', size=15)
    width, height = cm.shape
    a = [0, 0, 0, 0, 0]
    for i in range(len(a)):
        for j in range(len(a)):
            a[i] = cm[i][j] + a[i]
    for x in range(width):
        for y in range(height):
            plt.annotate(str(np.round(cm[x][y], 2)), xy=(y, x), horizontalalignment='center',verticalalignment='center')

    plt.savefig('/cptjack/totem/zhaofeiyan/DataSet/5-class/trained-model/val_1111.tif', bbox_inches='tight')

def draw_confusion_matrix():
    cm = confusion_matrix(test_labels, predictions)
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    plt.title('Confusion matrix', size=15)
    plt.colorbar()
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, ["0", "1", "3", "5", "7"], rotation=45, size=10)
    plt.yticks(tick_marks, ["0", "1", "3", "5", "7"], size=10)
    plt.tight_layout()
    plt.ylabel('Actual label', size=15)
    plt.xlabel('Predicted label', size=15)
    width, height = cm.shape
    a = [0, 0, 0, 0, 0]
    for i in range(len(a)):
        for j in range(len(a)):
            a[i] = cm[i][j] + a[i]
    for x in range(width):
        for y in range(height):
            plt.annotate(str(np.round(cm[x][y] / a[x], 2)), xy=(y, x), horizontalalignment='center',verticalalignment='center')

    plt.savefig('/cptjack/totem/zhaofeiyan/DataSet/5-class/trained-model/val_cell222_1111.tif', bbox_inches='tight')

plt.show()
draw_confusion_matrix_classes()
#draw_confusion_matrix()

