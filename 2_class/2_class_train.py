
"""
Created on 2019/10/24
@author: zfy

"""
import pandas as pd
import numpy as np
import argparse
import datetime
import random
import keras
import glob
import time
import sys
import os
from keras.models import *
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,MaxPooling2D,Flatten,BatchNormalization
from keras.callbacks import CSVLogger,ModelCheckpoint,ReduceLROnPlateau
from sklearn.metrics import classification_report,confusion_matrix
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from clssification_models.resnet import ResNet18
from keras.initializers import orthogonal
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize
from keras import backend as K
from keras import optimizers
from skimage import io
from keras.callbacks import CSVLogger,Callback,EarlyStopping
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.utils import multi_gpu_model
import h5py
from sklearn.model_selection import StratifiedKFold
from scipy import misc
from sklearn.model_selection import train_test_split
import gc
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input,decode_predictions
from PIL import Image
import xlrd
from sklearn.preprocessing import LabelBinarizer

os.environ['CUDA_VISIBLE_DEVICES']='0'

model = ResNet50(weights = None,include_top = False,pooling = 'max', input_shape = (224,224,3))
model.load_weights('/cptjack/sys_software_bak/keras_models/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

top_model = Sequential()
model.summary()
top_model.add(model)
#top_model.add(MaxPooling2D(pool_size=(2,2)))
#top_model.add(Flatten())
top_model.add(Dropout(0.5))
top_model.add(Dense(64,activation = 'relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2,activation='sigmoid',kernel_initializer=orthogonal()))

for layer in top_model.layers:
    layer.trainable = True

LearningRate = 0.00001
decay = 0.0001
n_epochs = 45

sgd = optimizers.SGD(lr=LearningRate, decay=LearningRate/n_epochs,momentum = 0.9, nesterov = True)
top_model.compile(optimizer = sgd, loss = 'binary_crossentropy',metrics=['accuracy'])
trainable_params = int(np.sum([K.count_params(p)for p in set(top_model.trainable_weights)]))
non_trainable_params = int(np.sum([K.count_params(p) for p in set(top_model.non_trainable_weights)]))

print("\nmodel Stats")
print("="*30)
print("Total Parameters:{:,}".format((trainable_params+non_trainable_params)))
print("Non-Trainable Parameters:{:,}".format(non_trainable_params))
print("Trainable Parameters:{:,}\n".format(trainable_params))

train_folders = '/cptjack/totem/zhaofeiyan/DataSet/DataSet8/training/'
validation_folders = '/cptjack/totem/zhaofeiyan/DataSet/DataSet8/validation/'

img_width,img_height = 224,224
batch_size_for_generators = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
                                    horizontal_flip=True,
                                    vertical_flip=True)

train_generator = train_datagen.flow_from_directory(train_folders,
                                                    target_size = (img_width,img_height),
                                                    batch_size = 32,
                                                    class_mode = 'categorical')

validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_directory(validation_folders,
                                                            target_size=(img_width,img_height),
                                                            batch_size = 32,
                                                            class_mode = 'categorical')

nb_train_samples = sum([len(files)for root,dirs,files in os.walk(train_folders)])
nb_validation_samples = sum([len(files)for root,dirs,files in os.walk(validation_folders)])

class Mycbk(ModelCheckpoint):

    def __init__(self, model, filepath ,monitor = 'val_loss',mode='min', save_best_only=True):
        self.single_model = model
        super(Mycbk,self).__init__(filepath, monitor, save_best_only, mode)
    def set_model(self,model):
        super(Mycbk,self).set_model(self.single_model)

def get_callbacks(filepath,model,j,patience=6):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = Mycbk(model, '/cptjack/totem/zhaofeiyan/DataSet/DataSet8-h5/'+ filepath)
    file_dir = '/cptjack/totem/zhaofeiyan/DataSet/DataSet8-h5/log/'+ time.strftime('%Y_%m_%d',time.localtime(time.time()))
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    tb_log = TensorBoard(log_dir=file_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                patience=4, verbose=0,
                                mode='min', epsilon=-0.95,
                                cooldown=0, min_lr=1e-8)

    log_cv = CSVLogger('/cptjack/totem/zhaofeiyan/DataSet/DataSet8-h5/'+ time.strftime('%Y_%m_%d',time.localtime(time.time())) +'_log.csv', separator=',', append=True)
    return [es, msave,reduce_lr,tb_log,log_cv]


file_path = "ResNet50_1030.h5"
callbacks_s = get_callbacks(file_path,top_model,224,patience=5)

train_steps = nb_train_samples//batch_size_for_generators
valid_steps = nb_validation_samples//batch_size_for_generators

model.save_weights('resnet_dataset8-1030.h5')
top_model.fit_generator(generator=train_generator,epochs=n_epochs,
                        steps_per_epoch=train_steps,
                        validation_data=validation_generator,
                        validation_steps = valid_steps, callbacks=callbacks_s, verbose=1)
