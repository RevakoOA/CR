import urllib.request
from flask import Flask, render_template, jsonify, request
import skimage.feature
from sklearn.externals import joblib
import skimage.io
from skimage.transform import resize
import numpy as np
from skimage import color



app = Flask(__name__)

clf = joblib.load("data/digits_cls.pkl")


@app.route('/')
def index():
    """
    Uses Flask's Jinja2 template renderer to generate the html
    """
    return render_template('index.html')

@app.route('/predict/')
def predict():

    image_vector = request.args.get('image')

    image_vector = image_vector[1:len(image_vector) - 1]
    print(image_vector)
    image = urllib.request.urlretrieve(image_vector, filename='home/nikita/PycharmProjects/HandMade/res/image')

    print(image[0])

    image = np.array(color.rgb2gray(skimage.io.imread(image[0])))
    image = resize(image=image, output_shape=(28, 28))

    clf = joblib.load("data/digits_cls.pkl")

    hog = skimage.feature.hog(image, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)

    return jsonify(str(clf.predict(hog)))

if __name__ == '__main__':
    app.run(debug=True)
