import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os import path
import pandas as pd
import xml.etree.ElementTree as xt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
# understand this and imagenet
from tensorflow.keras.applications import MobileNetV2, InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard


# read the CSV file using pandas
df = pd.read_csv('images-Set2/labels.csv')

def get_file_name(filename):
    filename = 'images-Set2/'+filename
    file_name_images = xt.parse(filename).getroot().find('filename').text
    file_path_images = os.path.join('./images-Set2/Images', file_name_images)
    return file_path_images


image_path = list(df['filepath'].apply(get_file_name))
# print(image_path)

labels = df.iloc[:,1:].values
# print(labels)

data = []
output = []

for i in range(len(image_path)):
    image = image_path[i]
    if path.exists(image):
        image_array = cv2.imread(image)
        w, h, d = image_array.shape
        loaded_img = load_img(image, target_size=(224, 224))
        loaded_img_array = img_to_array(loaded_img)
        normal_loaded_img_array = loaded_img_array / 255.0
        xmin, ymin, xmax, ymax = labels[i]
        nxmin, nymin = xmin / w, ymin / h
        nxmax, nymax = xmax / w, ymax / h
        normal_labels = (nxmin, nxmax, nymin, nymax)
        data.append(normal_loaded_img_array)
        output.append(normal_labels)
    else:
        continue



X = np.array(data, dtype=np.float32)
y = np.array(output, dtype=np.float32)
X_train,X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

inception_resnet = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
inception_resnet.trainable = False
end_of_model = inception_resnet.output
end_of_model = Flatten()(end_of_model)
end_of_model = Dense(800, activation='relu')(end_of_model)
end_of_model = Dense(500, activation='relu')(end_of_model)
end_of_model = Dense(250, activation='relu')(end_of_model)
end_of_model = Dense(4, activation='sigmoid')(end_of_model)
model = Model(inputs=inception_resnet.input, outputs=end_of_model)
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

# tb = TensorBoard('object_detection_set2')
# history = model.fit(x=X_train, y=y_train, batch_size=32, epochs=400, validation_data=(X_test,y_test), callbacks=[tb])
history = model.fit(x=X_train, y=y_train, batch_size=32, epochs=180, validation_data=(X_test,y_test))

model.save('Models/Set2_object_detection_William_29_11_2021.h5')



