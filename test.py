import sys
import os
from Ui_test_ui import Ui_Dialog
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtGui
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
# import random

from keras.models import Model, load_model, model_from_json, Sequential
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

class MainWindow(QMainWindow, Ui_Dialog):    
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    def onBindingUI(self):
        self.pushButton.clicked.connect(self.on_btn1_click)
        self.pushButton_2.clicked.connect(self.on_btn2_click)

    # If button 1 click
    def on_btn1_click(self):
        # File dialog
        root = tk.Tk()
        root.withdraw()
        global source_path
        source_path = filedialog.askopenfilename()

        #  Source label
        sourcelabel_img=QtGui.QPixmap(source_path).scaled(self.label_8.width(), self.label_8.height())
        self.label_8.setPixmap(sourcelabel_img)

        return 

    # If button 2 click
    def on_btn2_click(self):
        # File dialog
        root = tk.Tk()
        root.withdraw()
        label_path = filedialog.askopenfilename()

        # Source label
        sourcegroundtrue_img=QtGui.QPixmap(label_path).scaled(self.label_9.width(), self.label_9.height())
        self.label_9.setPixmap(sourcegroundtrue_img)

        # Image height, wedth
        img_groundtrue = cv2.imread(label_path, 0)
        img_h, img_w = img_groundtrue.shape

        # Count groungtrue pixel number
        groundtrue_pixel = 0
        for i in range(1, img_h - 1):
            for j in range(1, img_w - 1):
                if img_groundtrue [i, j] > 50:
                    groundtrue_pixel = groundtrue_pixel + 1
        
        # Load source image
        X = np.zeros((1, 512, 512, 1), dtype=np.float32)
        img = load_img(source_path, color_mode = "grayscale")
        x_img = img_to_array(img)
        X[0, ..., 0] = x_img.squeeze() / 255
        X[0] = x_img / 255

        # Load model
        model = Sequential()
        model = load_model('model_4.h5')
        preds_train = model.predict(X, verbose=1)

        # Count result pixel number & count intersect region
        result_pixel = 0
        intersect_region = 0
        for i in range(1, 512 - 1):
            for j in range(1, 512 - 1):
                if preds_train[0][i, j] > 0.196:
                    preds_train[0][i, j] = 1
                    result_pixel = result_pixel + 1
                if (preds_train[0][i, j] > 0.196) & (img_groundtrue[i, j] > 50):
                    intersect_region = intersect_region + 1
        
        # Calculation Dice coefficient(DC)
        sum = groundtrue_pixel + result_pixel
        DC = (intersect_region*2)/sum

        # Show the result
        groundtrue_pixel = str(groundtrue_pixel)
        self.label_5.setText("Ground truth(pixel) : " + groundtrue_pixel)
        result_pixel = str(result_pixel)
        self.label_4.setText("Result(pixel):" + result_pixel)
        intersect_region = str(intersect_region)
        self.label_7.setText("Intersection(pixel) : " + intersect_region)
        DC = str(DC)
        self.label_6.setText("DC : " + DC)

        # Show the result image
        cv2.imshow("Result Imgage", preds_train[0]) 
        cv2.waitKey (0)
        cv2.destroyAllWindows()

        return
       
    ### ### ###

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


