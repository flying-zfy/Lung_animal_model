import numpy as np
import os
from keras.models import *
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D, Flatten,BatchNormalization
import cv2
import time
#from keras.applications.nasnet import NASNetMobile
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.initializers import Orthogonal,lecun_normal
from keras import backend as K
from keras import optimizers
from keras.callbacks import CSVLogger, Callback, EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
from keras.utils.np_utils import to_categorical

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ['CUDA_VISIBLE_DEVICES']='0'

model = InceptionV3(weights = None, include_top = False, pooling='max', input_shape = (299,299,3))
model.load_weights('/cptjack/sys_software_bak/keras_models/models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

top_model = Sequential()
top_model.add(model)
top_model.add(Dense(64, activation= 'selu', kernel_initializer=Orthogonal()))
top_model.add(Dropout(0.5))
#top_model.add(BatchNormalization())
top_model.add(Dense(64, activation= 'selu'))
top_model.add(Dropout(0.5))
#top_model.add(BatchNormalization())
top_model.add(Dense(5, activation='softmax',kernel_initializer=Orthogonal()))
top_model.summary()

for layer in model.layers:
    layer.trainable = True

LearningRate = 0.00005
decay = 0.0001
n_epochs = 60

sgd = optimizers.SGD(lr=LearningRate, decay=LearningRate/n_epochs, momentum=0.9, nesterov=True)
top_model.compile(optimizer = sgd,loss = 'categorical_crossentropy',metrics=['accuracy'])
trainable_params = int(np.sum([K.count_params(p) for p in set(top_model.trainable_weights)]))
non_trainable_params = int(np.sum([K.count_params(p) for p in set(top_model.non_trainable_weights)]))

print("\nModel Stats")
print("=" * 30)
print("Total Parameters: {:,}".format((trainable_params + non_trainable_params)))
print("Non-Trainable Parameters: {:,}".format(non_trainable_params))
print("Trainable Parameters: {:,}\n".format(trainable_params))

train_folders = '/cptjack/totem/zhaofeiyan/Data-new/Data1/tra/'
validation_folders = '/cptjack/totem/zhaofeiyan/Data-new/Data1/val/'

img_width,img_height = 299,299
batch_size_for_generators = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=True,)

train_generator = train_datagen.flow_from_directory(train_folders,
                                                    target_size = (img_width,img_height),
                                                    batch_size = batch_size_for_generators,
                                                    class_mode = 'categorical')

validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_directory(validation_folders,
                                                              target_size=(img_width,img_height),
                                                              batch_size = batch_size_for_generators,
                                                              class_mode = 'categorical')
print(train_generator.class_indices)
print("\n")

nb_train_samples = sum([len(files)for root,dirs,files in os.walk(train_folders)])
nb_validation_samples = sum([len(files)for root,dirs,files in os.walk(validation_folders)])


class Mycbk(ModelCheckpoint):

    def __init__(self, model, filepath ,monitor = 'val_loss',mode='min', save_best_only=True):
        self.single_model = model
        super(Mycbk,self).__init__(filepath, monitor, save_best_only, mode)
    def set_model(self,model):
        super(Mycbk,self).set_model(self.single_model)

def get_callbacks(filepath,model,patience=6):

    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = Mycbk(model, '/cptjack/totem/zhaofeiyan/Data-new/Data1-h/inception/'+filepath)
    file_dir = '/cptjack/totem/zhaofeiyan/Data-new/Data1-h/inception/log/'+ time.strftime('%Y_%m_%d',time.localtime(time.time()))
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    tb_log = TensorBoard(log_dir=file_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=4, verbose=0, mode='min',
                                  epsilon=-0.95, cooldown=0, min_lr=1e-8)
    log_cv = CSVLogger('/cptjack/totem/zhaofeiyan/Data-new/Data1-h/inception/' + time.strftime('%Y_%m_%d',time.localtime(time.time())) +'_log.csv', separator=',', append=True)
    return [es, msave,reduce_lr,tb_log,log_cv]


file_path = "Inception_Mobile_1115.h5"
callbacks_s = get_callbacks(file_path,top_model,patience=10)

train_steps = nb_train_samples//batch_size_for_generators
valid_steps = nb_validation_samples//batch_size_for_generators

model.save_weights('inception_1115.h5')
top_model.fit_generator(train_generator,epochs=n_epochs,
                        steps_per_epoch=train_steps,validation_data=validation_generator,
                        validation_steps = valid_steps, callbacks=callbacks_s, verbose=1)
