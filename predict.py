from tensorflow.keras.models import load_model
from tensorflow.nn import softmax
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow import expand_dims
import numpy as np
import pickle

# load model
model_path = 'model_0.hdf5'
model = load_model(model_path)


# load label dictionary
label_path = 'model_0_labels.pickle'
with open(label_path, 'rb+') as f:
    label_dict = pickle.load(f)

batch_size = 32
img_height = 180
img_width = 180

test_path = "/mnt/c/Users/mathe/work/data/flower_photos/sunflowers/2979133707_84aab35b5d.jpg"
img = load_img(test_path, target_size=(img_height, img_width))
img_array = img_to_array(img)
img_array = expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(label_dict[np.argmax(score)], 100 * np.max(score))
)