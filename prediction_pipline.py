import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import pytesseract as pt


def plate_detection(path,model):
    image = load_img(path)
    image = np.array(image, dtype=np.uint8)
    image_resize = load_img(path, target_size=(224, 224))
    image_array_normalized = img_to_array(image_resize) / 255.0
    w, h, d = image.shape
    test_array = image_array_normalized.reshape(1, 224, 224, 3)
    coords = model.predict(test_array)
    denorm = np.array([w, w, h, h])
    original_coords = (coords * denorm).astype(np.int32)
    xmin, xmax, ymin, ymax = original_coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    cv2.rectangle(image, pt1, pt2, color=(0, 255, 255), thickness=2)
    return image, original_coords

def main():
    model = load_model('Models/Set2_object_detection_William_17_07_2021.h5')
    path = 'Test_Images/Hello.png'
    image, cords = plate_detection(path, model)
    plt.figure()
    plt.imshow(image)
    plt.show()
    img = np.array(load_img(path))
    print(img)
    xmin, xmax, ymin, ymax = cords[0]
    print(xmin, xmax, ymin, ymax)
    roi = img[ymin:ymax, xmin:xmax]
    plt.imshow(roi)
    plt.show()
    plate_number = pt.image_to_string(path)
    print(plate_number)


if __name__ == '__main__':
    main()

