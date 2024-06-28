#imports
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import seaborn as sns

#Data loading and initialization
image_size = 64         #setting image_size to 64 instead of 768 for faster computation time
lung_benign_class = 0
lung_aca_class = 1          #creating our 5 different classes
lung_scc_class=2
colon_benign_class=3
colon_aca_class=4
disease_class = ['lungbenign', 'lungaca','lungscc','colonbenign','colonaca']


lung_benign_file_folder = "C:/Users/NIKOS/PycharmProjects/lung_colon_image_set/lung_image_sets/lung_n"
lung_benign_files = glob.glob(os.path.join(lung_benign_file_folder, "*"))

lung_aca_file_folder = "C:/Users/NIKOS/PycharmProjects/lung_colon_image_set/lung_image_sets/lung_aca"
lung_aca_files = glob.glob(os.path.join(lung_aca_file_folder, "*"))

lung_scc_file_folder = "C:/Users/NIKOS/PycharmProjects/lung_colon_image_set/lung_image_sets/lung_scc"
lung_scc_files = glob.glob(os.path.join(lung_scc_file_folder, "*"))

colon_benign_file_folder = "C:/Users/NIKOS/PycharmProjects/lung_colon_image_set/colon_image_sets/colon_n"
colon_benign_files = glob.glob(os.path.join(colon_benign_file_folder, "*"))

colon_aca_file_folder = "C:/Users/NIKOS/PycharmProjects/lung_colon_image_set/colon_image_sets/colon_aca"
colon_aca_files = glob.glob(os.path.join(colon_aca_file_folder, "*"))

#create a list of the labels
labels=list()

#plot a 4x4 grid of subplots for the lung_n directory
lung_benign_images = np.zeros((len(lung_benign_files), image_size, image_size, 3))

for no, name in enumerate(lung_benign_files):
    pil_img = image.load_img(path=name, color_mode="rgb", target_size=(image_size, image_size))

    lung_benign_images[no, :, :, :] = image.img_to_array(pil_img)
    labels.append(lung_benign_class)

fig, ax = plt.subplots(4, 4, figsize=(9, 9))
n = 0
for i in range(4):
    for j in range(4):
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].imshow(image.load_img(path=lung_benign_files[n], color_mode="rgb"))
        n += 1
plt.tight_layout()
plt.show()

#plot a 4x4 grid of subplots for the lung_aca directory
lung_aca_images = np.zeros((len(lung_aca_files), image_size, image_size, 3))

for no, name in enumerate(lung_aca_files):
    pil_img = image.load_img(path=name, color_mode="rgb", target_size=(image_size, image_size))

    lung_aca_images[no, :, :, :] = image.img_to_array(pil_img)
    labels.append(lung_aca_class)

fig, ax = plt.subplots(4, 4, figsize=(9, 9))
n = 0
for i in range(4):
    for j in range(4):
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].imshow(image.load_img(path=lung_aca_files[n], color_mode="rgb"))
        n += 1
plt.tight_layout()
plt.show()

#plot a 4x4 grid of subplots for the lung_scc directory
lung_scc_images = np.zeros((len(lung_scc_files), image_size, image_size, 3))

for no, name in enumerate(lung_scc_files):
    pil_img = image.load_img(path=name, color_mode="rgb", target_size=(image_size, image_size))

    lung_scc_images[no, :, :, :] = image.img_to_array(pil_img)
    labels.append(lung_scc_class)

fig, ax = plt.subplots(4, 4, figsize=(9, 9))
n = 0
for i in range(4):
    for j in range(4):
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].imshow(image.load_img(path=lung_scc_files[n], color_mode="rgb"))
        n += 1
plt.tight_layout()
plt.show()

#plot a 4x4 grid of subplots for the colon_n directory
colon_benign_images = np.zeros((len(colon_benign_files), image_size, image_size, 3))

for no, name in enumerate(colon_benign_files):
    pil_img = image.load_img(path=name, color_mode="rgb", target_size=(image_size, image_size))

    colon_benign_images[no, :, :, :] = image.img_to_array(pil_img)
    labels.append(colon_benign_class)

fig, ax = plt.subplots(4, 4, figsize=(9, 9))
n = 0
for i in range(4):
    for j in range(4):
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].imshow(image.load_img(path=colon_benign_files[n], color_mode="rgb"))
        n += 1
plt.tight_layout()
plt.show()

#plot a 4x4 grid of subplots for the colon_aca directory
colon_aca_images = np.zeros((len(colon_aca_files), image_size, image_size, 3))

for no, name in enumerate(colon_aca_files):
    pil_img = image.load_img(path=name, color_mode="rgb", target_size=(image_size, image_size))

    colon_aca_images[no, :, :, :] = image.img_to_array(pil_img)
    labels.append(colon_aca_class)

fig, ax = plt.subplots(4, 4, figsize=(9, 9))
n = 0
for i in range(4):
    for j in range(4):
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].imshow(image.load_img(path=colon_aca_files[n], color_mode="rgb"))
        n += 1
plt.tight_layout()
plt.show()
#convert labels into an one-hot encoded array
labels = to_categorical(labels)

#concatenate all the images into a new array called images

images = np.concatenate((lung_benign_images,lung_aca_images,lung_scc_images,colon_benign_images,colon_aca_images), axis=0)
files = lung_benign_files + lung_aca_files + lung_scc_files + colon_benign_files + colon_aca_files

#split into train,test,and validation set
index = np.arange(len(images))
train, test = train_test_split(index, test_size=0.2, shuffle=True, random_state=0)

train, val = train_test_split(train, test_size=0.1, shuffle=True, random_state=0)

#create seperate datasets for training validation and testing

X_train = images[train,:,:,:]
X_val = images[val,:,:,:]
X_test = images[test,:,:,:]

labels = np.array(labels)
y_train = labels[train,:]
y_val = labels[val,:]
y_test = labels[test,:]

files = np.array(files)
f_train = files[train]
f_val = files[val]
f_test = files[test]

#normalising the pixel values of the images, dividing them by 255

X_train = X_train / 255
X_val = X_val / 255
X_test = X_test / 255

#cnn model creation

input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

cnn_model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", input_shape=input_shape),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dropout(rate=0.2),
    Dense(units=128, activation="relu"),
    Dropout(rate=0.2),
    Dense(units=5, activation="softmax"),
])

cnn_model.summary()
optimizer = Adam(learning_rate = 0.002)
cnn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#callbacks

annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
checkpoint = ModelCheckpoint('best_model.h5', verbose=1, save_best_only=True)

#fit and train the model
history = cnn_model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=20,
    verbose=1,
    callbacks=[annealer, checkpoint],
    validation_data=(X_val, y_val)
)

#plot training and validation categorical cross-entropy (CE) loss values
#plot training and validation accuracy

fig, axis = plt.subplots(nrows=1,ncols=2, figsize=(12,4))

axis[0].plot(history.epoch, history.history['loss'])
axis[0].plot(history.epoch, history.history['val_loss'])
axis[0].set_xlabel("Epochs")
axis[0].set_ylabel("Value")
axis[0].legend(["CE", "Val_CE"])
axis[0].set_title("Training Process - CE")

axis[1].plot(history.epoch, history.history['accuracy'])
axis[1].plot(history.epoch, history.history['val_accuracy'])
axis[1].set_xlabel("Epochs")
axis[1].set_ylabel("Value")
axis[1].legend(["Accuracy", "Val_Accuracy"])
axis[1].set_title("Training Process - Accuracy")
plt.show()


#predict for one image
image_path="C:/Users/NIKOS/PycharmProjects/lung_colon_image_set/lung_image_sets/lung_scc/lungscc1.jpeg"
image_to_predict=image.load_img(path=image_path, color_mode="rgb", target_size=(image_size, image_size))
img_array = image.img_to_array(image_to_predict) / 255.0  # Normalize the pixel values
image_to_predict = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size

model=load_model('best_model.h5')
prediction_probs = model.predict(image_to_predict)
predicted_class = np.argmax(prediction_probs)
predicted_label = disease_class[predicted_class]


plt.imshow(image_to_predict[0])
plt.title("Predicted Label: " + predicted_label)
plt.axis('off')
plt.show()




