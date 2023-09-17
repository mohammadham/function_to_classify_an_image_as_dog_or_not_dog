import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# function to classify an image as dog or not dog
def classify_dog(image_path):
    # load the image
    img = image.load_img(image_path, target_size=(224, 224))
    # preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # predict the class probabilities
    preds = model.predict(x)
    # decode the predictions
    decoded_preds = decode_predictions(preds, top=1)[0]
    # check if the top prediction is for a dog
    if decoded_preds[0][1] == 'dog':
        return True
    else:
        return False