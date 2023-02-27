import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import pytesseract as pt
import argparse
import math
import imutils


# batch_normalization(research), np.uint8

model = load_model('Models/Set3_object_detection_William_02_10_2021.h5')
print("Model is loaded succesfully")
# print(model.summary())

path = 'Test_Images/morton.jpg'

image = load_img(path)
image = np.array(image, dtype=np.uint8)
# print(image)
# print(image.shape)
image_resize = load_img(path, target_size=(224, 224))

image_array_normalized = img_to_array(image_resize) / 255.0
# this need to change to h,w,d
w, h, d = image.shape
# print(w,h,d)

plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.show()

test_array = image_array_normalized.reshape(1, 224, 224, 3)
print(test_array)

coords = model.predict(test_array)
denorm = np.array([w, w, h, h])
original_coords = (coords * denorm).astype(np.int32)
# print(denorm)
print(coords)
print(original_coords)

xmin, xmax, ymin, ymax = original_coords[0]
pt1 = (xmin, ymin)
pt2 = (xmax, ymax)

cv2.rectangle(image, pt1, pt2, color=(0, 255, 255), thickness=1)
plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.show()

roi = image[ymin:ymax, xmin:xmax]
print(roi)


def skew_correction():
    img = cv2.imread(roi)
    size = img.shape
    inverted_img = cv2.bitwise_not(img, img)
    edges = cv2.Canny(inverted_img, 50, 200, None, 3)
    edges_t = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_np = np.copy(edges_t)
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, None, size[1] / 2, 20)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(edges_np, (l[0], l[1]), (l[2], l[3]), color=(255, 0, 0), thickness=3)
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(edges_t)
    plt.imshow(edges_np)
    plt.show()

    angle = 0
    n_lines = len(linesP)
    display_lines = []
    display_lines = np.float32(display_lines)
    for i in range(0, n_lines):
        l = linesP[i][0]
        cv2.line(display_lines, (l[0], l[1]), (l[2], l[3]), (255, 0, 0))
        angle += math.atan2(l[3] - l[1], l[2] - l[0])
    print(angle)
    rotated_roi = imutils.rotate(img, angle)
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2RGB))
    plt.show()
    return rotated_roi


gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
filter_roi = cv2.bilateralFilter(gray, 60, 10, 10)

figure = plt.figure()
figure.add_subplot(2, 1, 1)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

figure.add_subplot(2, 1, 2)
plt.imshow(cv2.cvtColor(filter_roi, cv2.COLOR_BGR2RGB))

plt.show()


def variants_of_Laplacian(img):
    var_num = cv2.Laplacian(img, cv2.CV_64F).var()
    return var_num


def arg_parse():
    ag = argparse.ArgumentParser()
    ag.add_argument("-i", "--images", required=True, help="path to input directory")
    ag.add_argument("-t", "--threshold", type=float, default=100.00,
                    help="focus that falls below this value will considered blury")
    args = vars(ag.parse_args())
    return args


fm = variants_of_Laplacian(filter_roi)

print("Laplacian variance is: ", fm)

threshold = 2000

if fm <= threshold:
    print("Not Blurry")
else:
    print("Blurry")


def main():
    skew_correction()


if __name__ == '__main__':
    main()
