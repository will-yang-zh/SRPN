import os
import glob
import cv2




path = 'images-Set1\\'

files = glob.glob(path+'/*'+'*.jpeg')
print(files)


def BW_maker(filename):

    img = cv2.imread(path+filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('images-Set1/BW_image/{}'.format(filename), gray)


for i in range(len(files)):

    filename = files[i].split('\\')[1]

    # print(filename)
    BW_maker(filename)

