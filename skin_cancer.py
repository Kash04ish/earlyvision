import cv2
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import math
import shutil
import glob

image_dir = r"Skin cancer ISIC The International Skin Imaging Collaboration"
metadata = []
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):  
        image_path = os.path.join(image_dir, filename)
        img = cv2.imread(image_path)
        height, width, channels = img.shape
        metadata.append({'Filename': filename, 'Width': width, 'Height': height, 'Channels': channels})

df = pd.DataFrame(metadata)

print(df)

root = "Skin cancer ISIC The International Skin Imaging Collaboration\Train"
# noiTotal = {}
# for dir in os.listdir(root):
#     noiTotal[dir] = len(os.listdir(os.path.join(root, dir)))

# print(noiTotal.items())

def splitter(p,split):
    if not os.path.exists("./"+p):
        os.mkdir("./"+p)
        for dir in os.listdir(root):
            os.makedirs("./"+p+"/"+dir)
            for img in np.random.choice(a = os.listdir(os.path.join(root,dir)),size = max(math.floor(split * noiTotal[dir]) - 5, 0),replace=False):
                O = os.path.join(root,dir,img)
                D = os.path.join('./'+p,dir)
                shutil.copy(O,D)
                os.remove(O)
    else:
        print(f"{p}Floder exist")

splitter("Validation", 0.15)
splitter("Test", 0.15)

# Test = "Test"
# noiTest = {}
# for dir in os.listdir(root):
#     noiTest[dir] = len(os.listdir(os.path.join(Test, dir)))

# print(noiTest.items())

# Train = "Skin cancer ISIC The International Skin Imaging Collaboration\Train"
# noiTrain = {}
# for dir in os.listdir(root):
#     noiTrain[dir] = len(os.listdir(os.path.join(Train, dir)))

# print(noiTrain.items())

# Validation = "Validation"
# noiValidation = {}
# for dir in os.listdir(root):
#     noiValidation[dir] = len(os.listdir(os.path.join(Validation, dir)))

# print(noiValidation.items())

#################################BUILDING MODEL#####################################

import keras
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
# from keras.applications.mobilenet import MobileNet, preprocess_input

# CNN
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))

model.add(Conv2D(filters=36, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))  

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))  

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=9, activation='softmax'))

# model.summary()
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

def preprocessingImages1(path):
    image_data = ImageDataGenerator(zoom_range=0.2, shear_range=0.2, rescale=1/255, horizontal_flip=True)
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='categorical')

    return image

path = "Skin cancer ISIC The International Skin Imaging Collaboration\Train"
train_data = preprocessingImages1(path)

def preprocessingImages2(path):
    image_data = ImageDataGenerator(rescale=1/255)
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='categorical')

    return image

path = "Test"
test_data = preprocessingImages2(path)

path = "Validation"
Validation_data = preprocessingImages2(path)


# base_model = MobileNet(input_shape=(224, 224, 3), include_top=False)
# for layer in base_model.layers:
#     layer.trainable = False

# x = Flatten()(base_model.output)
# x = Dense(units=1, activation='softmax')(x)

# model = model(base_model.input, x)
# model.compile(optimizer='rnsprop', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


from keras.callbacks import ModelCheckpoint, EarlyStopping

es = EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=5, verbose=1, mode='auto')

mc = ModelCheckpoint(monitor="val_accuracy", filepath="./bestmodel.keras", verbose=1, save_best_only=True, mode='auto')
cd=[es,mc]

#################MODEL TRAINING#############################

# hs=model.fit_generator(train_data,steps_per_epoch=8,epochs=30,verbose=1,validation_data=Validation_data,validation_steps=16,callbacks=cd)

hs = model.fit(
    train_data,  
    steps_per_epoch=8,  
    epochs=50,  
    verbose=1,  
    validation_data=Validation_data, 
    validation_steps=16,  
    callbacks=cd  
    )

import matplotlib.pyplot as plt

h = hs.history
# dict_keys = h.keys() 
# dict_keys(['loss', 'accuracy', 'validation_loss', 'validation_accuracy'])

# print(h.keys())

plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'], c='red')
plt.title("accuracy vs validation-accuracy")
plt.show()


plt.plot(h['loss'])
plt.plot(h['val_loss'], c='red')
plt.title("loss vs validation-loss ")
plt.show()

##############MODEL ACCURACY#################################

from keras.models import load_model

model = load_model("bestmodel.keras")

acc = model.evaluate(test_data)[1]
# acc = model.evaluate_generator(test_data)[1]
print(f"Accuracy of model is: {acc*100}%")

from keras.preprocessing.image import load_img, img_to_array

path = "Test/melanoma/ISIC_0000022.jpg"

img = load_img(path, target_size=(224, 224))
input_arr = img_to_array(img)/255

input_arr = np.expand_dims(input_arr, axis=0)

# pred = model.predict_classes(input_arr)[0][0]
pred_probabilities = model.predict(input_arr)
pred_class = np.argmax(pred_probabilities)

print(pred_class)
print(train_data.class_indices)

if pred_class==0:
    print("You have Skin Cancer and its type is Actinic Keratosis.")
elif pred_class==1:
    print("You have Skin Cancer and its type is Basal cell carcinoma.")
elif pred_class==2:
    print("You have Skin Cancer and its type is Dermatofibroma.")
elif pred_class==3:
    print("You have Skin Cancer and its type is Melanoma.")
elif pred_class==4:
    print("You have Skin Cancer and its type is Nevus.")
elif pred_class==5:
    print("You have Skin Cancer and its type is Pigmented benign keratosis.")
elif pred_class==6:
    print("You have Skin Cancer and its type is Seborrheic keratosis.")
elif pred_class==7:
    print("You have Skin Cancer and its type is Squamous cell carcinoma.")
elif pred_class==8:
    print("You have Skin Cancer and its type is Vascular lesion.")
else:
    print("Your sample did not match with any type of Skin Cancer. The probability of having skin cancer is almost 0.")


