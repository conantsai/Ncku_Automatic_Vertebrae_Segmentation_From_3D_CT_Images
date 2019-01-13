import os
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import tensorflow as tf

from keras.models import Model, load_model, model_from_json
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

image_width, image_height, image_channel = 512, 512, 1
path_traindata = './test/data3/train'
path_testdata = './test/data3/test'

train_ids = []
test_ids = []

# Get traindata name
for dirPath, dirNames, fileNames in os.walk(path_traindata + '/image/'):
  for f in fileNames:
    train_ids.append(f)

# Get testdata name
for dirPath, dirNames, fileNames in os.walk(path_testdata + '/image/'):
  for f in fileNames:
    test_ids.append(f)

X = np.zeros((len(train_ids), image_height, image_width, image_channel), dtype=np.float32)
y = np.zeros((len(train_ids), image_height, image_width, image_channel), dtype=np.bool)
X_val = np.zeros((len(test_ids), image_height, image_width, image_channel), dtype=np.float32)
y_val = np.zeros((len(test_ids), image_height, image_width, image_channel), dtype=np.bool)

# Get and resize train images and label
def get_data(X, y, ids, path):
    print('Getting and resizing images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + '/image/' + id_, color_mode = "grayscale")
        x_img = img_to_array(img)
        # x_img = resize(x_img, (512, 512, 1), mode='constant', preserve_range=True)

        # Load label
        label = img_to_array(load_img(path + '/label/' + id_, color_mode = "grayscale"))
        # label= resize(label, (512, 512, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        X[n] = x_img / 255
        y[n] = label / 255

    return X, y
    
# Get train data & valid data
X_train, y_train = get_data(X, y, train_ids, path_traindata)
X_valid, y_valid = get_data(X_val, y_val, test_ids, path_testdata)

def conv2d_block(input_tensor, filters, kernel_size=3, batchnorm=True):
    conv = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm:
        conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    
    return conv

# Define U-net
def get_unet(input_img, filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, filters=filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, filters=filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, filters=filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, filters=filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, filters=filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, filters=filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, filters=filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, filters=filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, filters=filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

input_img = Input((image_height, image_width, image_channel), name='img')

model = get_unet(input_img, filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# EarlyStopping
early_stopping = EarlyStopping(patience=10, verbose=1) 
# ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1) 
# ModelCheckpoint
model_checkpoint = ModelCheckpoint('model_4.h5', verbose=1, save_best_only=True, save_weights_only=False) 

results = model.fit(X_train, y_train, batch_size=4, epochs=50, callbacks=[early_stopping, reduce_lr, model_checkpoint], validation_data=(X_valid, y_valid))







