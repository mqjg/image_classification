from tensorflow.keras.models import load_model
from tensorflow.nn import softmax
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow import expand_dims
import numpy as np
import pickle
import argparse

class ImageClassifier:
    
    def __init__(self, model_path, labels_path):
        # load model
        self.model = load_model(model_path)

        # load labels
        with open(labels_path, 'rb+') as f:
            self.labels = pickle.load(f)

        # image sizes hardcoded to model, move these to yaml file!
        self.img_height, self.img_width = (180, 180)

    def predict(self, img_path):
        img = load_img(img_path, 
                       target_size=(self.img_height, self.img_width))
        img_array = img_to_array(img)
        img_array = expand_dims(img_array, 0) # Create a batch

        predictions = self.model.predict(img_array)
        score = softmax(predictions[0])

        return self.labels[np.argmax(score)], np.max(score)

if __name__ == '__main__':
    model_path = 'model_0.hdf5'
    labels_path = 'model_0_labels.pickle'
    model = ImageClassifier(model_path, labels_path)

    img_path = "../data/flower_photos/sunflowers/2979133707_84aab35b5d.jpg"
    label, score = model.predict(img_path)

    print(f"Model predicted \"{label}\" with a confidence of {score: .3f}")