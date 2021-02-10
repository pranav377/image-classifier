import numpy as np 
import cv2
import os
import pickle

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras
from keras import layers, models
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model


class CreateDataAndModel():

    def __init__(self, file_path, model_file_name_to_save, init_lr=0.0001, epochs=100, batch_size=32):

        self.file_path = file_path
        self.INIT_LR = init_lr
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.model_name = model_file_name_to_save

    def create(self):

        self.le = LabelEncoder()
        self.label_dirs = os.listdir(self.file_path)

        self.data = []
        self.un_processed_labels = []

        for self.directory in self.label_dirs:

            self.img_data_file_path = (self.file_path + self.directory)
            self.files = os.listdir(self.img_data_file_path)

            for self.file in self.files:
                if self.file.endswith(".gif"):
                    print(self.img_data_file_path + "/" + self.file + " is a GIF file. GIF files cannot be used to create data. \n REMOVING GIF FILE FROM THE LIST...")
                    self.files.pop(self.files.index(self.file))

            for self.file in self.files:
                self.img_file_path = self.img_data_file_path + "/" + self.file
                self.img_data = cv2.imread(self.img_file_path)
                self.img_data = np.resize(self.img_data, (256, 256, 3))

                self.data.append(self.img_data)
                self.un_processed_labels.append(self.directory)

        self.data = np.array(self.data)
        self.un_processed_labels = np.array(self.un_processed_labels)
        self.labels = self.le.fit_transform(self.un_processed_labels)
        self.classes = self.le.classes_

        with open("./classes.pickle", "wb") as f:
            pickle.dump(self.classes, f)

        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(256, 256, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation="relu"))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation="relu"))
        self.model.add(layers.Dense(len(self.classes), activation="softmax"))

        self.opt = Adam(lr=self.INIT_LR, decay=self.INIT_LR / self.EPOCHS)

        self.model.compile(optimizer=self.opt, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        self.mc = ModelCheckpoint(self.model_name, monitor='accuracy', verbose=1, save_best_only=True, mode='max')

        self.model.fit(self.data, self.labels, epochs=self.EPOCHS, callbacks=[self.mc], batch_size=self.BATCH_SIZE)


class Run():

    def __init__(self, model_file_name):

        self.model_name = model_file_name
        self.model = load_model("./" + self.model_name)
        with open("./classes.pickle", "rb") as f:
            self.classes = pickle.load(f)

    def run(self, img_file_path):
        self.img_data = cv2.imread(img_file_path)
        self.img_data = np.resize(self.img_data, (1, 256, 256, 3))
        self.results = self.model.predict(self.img_data)
        self.pred = np.argmax(self.results)
        self.pred = self.classes[self.pred]

        return self.pred