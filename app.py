from flask import Flask, jsonify, request, redirect, render_template
import requests
import json
import csv
from flask_cors import CORS
from firebase import Firebase
import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
# CONVOLUTIONAL NETWORK
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib 
import pickle


data={}

# SE INICIALIZA FLASK
# app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')
# DA PERMISO DE CORS PARA INTERACCIÓN CON FRONTEND
CORS(app)


@app.route('/prueba')
def prueba():


    return ("pruena")



@app.route('/readData')
def readData():
    X_train=[]
    y_train=[]

    with open ('X_train', 'rb') as fp:
        X_train = pickle.load(fp)
        print(X_train.shape)
    
    with open ('y_train', 'rb') as fp:
        y_train = pickle.load(fp)
        print(y_train.shape)


    return (str(y_train.shape))


@app.route('/processImgTest')
def processimg():
    DATADIR = "chest_xray/test"

    CATEGORIES= ["NORMAL", "PNEUMONIA"]
    grayTrain=[]
    training_data=[]
    label_data=[]

    for category in CATEGORIES:
        path=os.path.join(DATADIR, category)
        class_num=CATEGORIES.index(category)
        # print(os.listdir(path))
        for img in os.listdir(path):
            originalImage=cv2.imread(os.path.join(path,img))
            grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
            (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
            img_array=blackAndWhiteImage
            new_array = cv2.resize(img_array, (100, 100))
            training_data.append([new_array])
            gray_new_array = cv2.resize(grayImage, (100, 100))
            grayTrain.append([gray_new_array])
            # original_new_array = cv2.resize(originalImage, (100, 100))
            # originalTrain.append([original_new_arrays])
            label_data.append([class_num])
    
    X_train=[]
    y_train=[]

    for image in grayTrain:
        X_train.append(image)
    
    X_train= np.array(X_train).reshape(-1,100,100)
    y_train = np.array(label_data)
    print(X_train.shape)
    print(y_train.shape)

    # plt.imshow(X_train[0,:,:], cmap=plt.cm.Greys)

    print(X_train[0, :, :])

    print(X_train.shape)

    ndims = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], ndims)

    y_train = y_train.reshape(-1,)

    print("Training Shape:", X_train.shape)
    print("Training Data Labels Shape:", y_train.shape)
    print(X_train[0])

    print("SCALER")
    scaler = MinMaxScaler().fit(X_train)

    X_train = scaler.transform(X_train)

    num_classes = 2
    y_train = to_categorical(y_train, num_classes)

    print(y_train[0])
    
    with open('X_test', 'wb') as fp:
        pickle.dump(X_train, fp)
    
    with open('y_test', 'wb') as fp:
        pickle.dump(y_train, fp)


    return ("Images Processed Test")


@app.route('/processImgTrain')
def processimgTrain():
    DATADIR = "chest_xray/train"

    CATEGORIES= ["NORMAL", "PNEUMONIA"]
    grayTrain=[]
    training_data=[]
    label_data=[]

    for category in CATEGORIES:
        path=os.path.join(DATADIR, category)
        class_num=CATEGORIES.index(category)
        # print(os.listdir(path))
        for img in os.listdir(path):
            originalImage=cv2.imread(os.path.join(path,img))
            grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
            (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
            img_array=blackAndWhiteImage
            new_array = cv2.resize(img_array, (100, 100))
            training_data.append([new_array])
            gray_new_array = cv2.resize(grayImage, (100, 100))
            grayTrain.append([gray_new_array])
            # original_new_array = cv2.resize(originalImage, (100, 100))
            # originalTrain.append([original_new_arrays])
            label_data.append([class_num])
    
    X_train=[]
    y_train=[]

    for image in grayTrain:
        X_train.append(image)
    
    X_train= np.array(X_train).reshape(-1,100,100)
    y_train = np.array(label_data)
    print(X_train.shape)
    print(y_train.shape)

    # plt.imshow(X_train[0,:,:], cmap=plt.cm.Greys)

    print(X_train[0, :, :])

    print(X_train.shape)

    ndims = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], ndims)

    y_train = y_train.reshape(-1,)

    print("Training Shape:", X_train.shape)
    print("Training Data Labels Shape:", y_train.shape)
    print(X_train[0])

    print("SCALER")
    scaler = MinMaxScaler().fit(X_train)

    X_train = scaler.transform(X_train)

    num_classes = 2
    y_train = to_categorical(y_train, num_classes)

    print(y_train[0])
    
    with open('X_train', 'wb') as fp:
        pickle.dump(X_train, fp)
    
    with open('y_train', 'wb') as fp:
        pickle.dump(y_train, fp)


    return ("Images Processed Train")


@app.route('/trainModel')
def trainModel():
    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]

    with open ('X_train', 'rb') as fp:
        X_train = pickle.load(fp)
        print(X_train.shape)
    
    with open ('y_train', 'rb') as fp:
        y_train = pickle.load(fp)
        print(y_train.shape)

    with open ('X_test', 'rb') as fp:
        X_test = pickle.load(fp)
        print(X_test.shape)
    
    with open ('y_test', 'rb') as fp:
        y_test = pickle.load(fp)
        print(y_test.shape)
    

    # convolutional network
    img_size = 100
    X_train = np.array(X_train).reshape(-1, img_size, img_size, 1)
    X_test= np.array(X_test).reshape(-1, img_size, img_size, 1)
    num_classes = 2

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])

    print(model.summary())

    model.fit(
        X_train,
        y_train,
        batch_size=128,
        epochs=10,
        shuffle=True,
        verbose=2
    )

    model.save('trainedModel.h5')
    

    return ("Entrenado y Guardado")

@app.route('/testModel')
def testModel():
    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]

    with open ('X_train', 'rb') as fp:
        X_train = pickle.load(fp)
        print(X_train.shape)
    
    with open ('y_train', 'rb') as fp:
        y_train = pickle.load(fp)
        print(y_train.shape)

    with open ('X_test', 'rb') as fp:
        X_test = pickle.load(fp)
        print(X_test.shape)
    
    with open ('y_test', 'rb') as fp:
        y_test = pickle.load(fp)
        print(y_test.shape)
    img_size = 100
    X_train = np.array(X_train).reshape(-1, img_size, img_size, 1)
    X_test= np.array(X_test).reshape(-1, img_size, img_size, 1)

    model = load_model('trainedModel.h5')
    predictions = model.evaluate(X_test,  y_test, verbose=2)

    return ("Accuray:"+str(predictions[1]))

@app.route("/upload-image", methods =["GET", "POST"])
def upload_image():
    if request.method =="POST":
        if request.files:
            image=request.files["image"]
            print(image)
        print("SUBIDO")
        return render_template('index.html')
    return render_template('index.html')
    


if __name__ == '__main__':
    app.run(debug=True, port=4000)