import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# بارگیری مدل شبکه عصبی سفارشی
model = tf.keras.models.load_model('my_custom_model.h5')

# تابع تشخیص سگ در تصویر
def dog_detector(image_path):
    # بارگیری تصویر با استفاده از کتابخانه keras.preprocessing.image
    image = load_img(image_path, target_size=(224, 224))
    # تبدیل تصویر به آرایه numpy
    image = img_to_array(image)
    # تغییر شکل آرایه به شکلی که شبکه عصبی مورد انتظار دارد
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # مقیاس‌دادن تصویر
    image = image.astype('float32') / 255.0
    # پیش‌بینی با استفاده از شبکه عصبی سفارشی
    prediction = model.predict(image)
    # بررسی آیا تصویر حاوی سگ است یا خیر
    if prediction[0] >= 0.5:
        return True
    else:
        return False

# تست تابع تشخیص سگ با چند تصویر
image_paths = ['test_dog.jpg', 'test_cat.jpg', 'test_bird.jpg']
for path in image_paths:
    if dog_detector(path):
        #return True
        print(f'{path} contains a dog')
    else:
        #return False
        print(f'{path} does not contain a dog')