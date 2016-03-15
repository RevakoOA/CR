from skimage.transform import resize
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.externals import joblib
import numpy as np
import skimage.io


five = np.array(color.rgb2gray(skimage.io.imread("res/five.jpg")))
three = np.array(color.rgb2gray(skimage.io.imread("res/three.jpg")))

#image = resize(image=image, output_shape=(28, 28))

clf = joblib.load("data/digits_cls.pkl")

hog_five = hog(five, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
hog_three = hog(three, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)

print(clf.predict(hog_five.reshape(1, -1)))

print(clf.predict(hog_three.reshape(1, -1)))
