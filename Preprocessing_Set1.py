import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
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
df = pd.read_csv('images-Set1/labels.csv')

print(df.head())
print(df.tail())


def get_file_name(filename):
    filename = 'images-Set1/'+filename
    file_name_images = xt.parse(filename).getroot().find('filename').text
    file_path_images = os.path.join('./images-Set1/BW_image', file_name_images)

    return file_path_images

# locating the file path to read from csv file
image_path = list(df['filepath'].apply(get_file_name))
print(image_path)
'''
# reading the image in using open cv
img = cv2.imread(image_path[0])

# displaying the image and the output
cv2.namedWindow('Sample Image', cv2.WINDOW_NORMAL)
cv2.imshow('Sample Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# create image with rectangle colour is BGR
cv2.rectangle(img, (1093, 645), (1396, 727), color=(0, 255, 0), thickness=3)
cv2.namedWindow('Rectangle Image', cv2.WINDOW_NORMAL)
cv2.imshow('Rectangle Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# preprocessing

# print(df.head)

# print(df.iloc[0])

# print(df.iloc[0,0])

labels = df.iloc[:,1:].values
# print(labels)

image = image_path[0]
image_array = cv2.imread(image)

# print(image_array
width,height,depth = image_array.shape
# print(width,height,depth)
loaded_img = load_img(image,target_size=(224,224))
# loaded_img_array = cv2.imread(loaded_img)
loaded_img_array = img_to_array(loaded_img)
# print(loaded_img_array)
# wid,hei,dep = loaded_img_array.shape
# print(wid,hei,dep)


# normalisation
normal_loaded_img_array = loaded_img_array/255.0
# print(normal_loaded_img_array)


xmin,ymin,xmax,ymax = labels[0]
nxmin,nymin = xmin/width, ymin/height
nxmax,nymax = xmax/width,ymax/height
normal_labels = (nxmin,nxmax,nymin,nymax)
# print(normal_labels)

data = []
output = []
'''
for i in range(len(image_path)):
    image = image_path[i]
    if os.path.exists(image):
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
'''

# print(data[0])
# print(output)

X = np.array(data, dtype=np.float32)
y = np.array(output, dtype=np.float32)
X_train,X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

inception_resnet = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
inception_resnet.trainable = False
end_of_model = inception_resnet.output
end_of_model = Flatten()(end_of_model)
end_of_model = Dense(500, activation='relu')(end_of_model)
end_of_model = Dense(250, activation='relu')(end_of_model)
end_of_model = Dense(4, activation='sigmoid')(end_of_model)
model = Model(inputs=inception_resnet.input, outputs=end_of_model)
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
# model.summary()
tb = TensorBoard('object_detection')
history = model.fit(x=X_train, y=y_train, batch_size=10, epochs=300, validation_data=(X_test,y_test), callbacks=[tb])
model.save('Models/Set1_object_detection.h5')




