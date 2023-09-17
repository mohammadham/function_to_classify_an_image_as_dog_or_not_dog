import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# بارگزاری مدل VGG-16 پیش‌آموزش‌شده
model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)

# تنظیمات اندازه تصویر و پیش‌پردازش
IMG_SIZE = (224, 224)
preprocess_input = tf.keras.applications.vgg16.preprocess_input

# تابع تشخیص سگ در تصویر
def predict_dog(image_path):
    # بارگزاری تصویر
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    # تبدیل تصویر به آرایه numpy
    x = tf.keras.preprocessing.image.img_to_array(img)
    # پیش‌پردازش تصویر
    x = preprocess_input(x)
    # تغییر ابعاد تصویر به شکل یک دسته
    x = np.expand_dims(x, axis=0)
    # پیش‌بینی با استفاده از شبکه‌ی VGG-16
    preds = model.predict(x)
    # تفسیر خروجی شبکه
    decoded_preds = tf.keras.applications.vgg16.decode_predictions(preds, top=1)[0][0]
    # در صورت تشخیص سگ، خروجی True و نام دسته‌بندی را برمی‌گرداند، در غیر این صورت خروجی False را برمی‌گرداند
    if decoded_preds[1] == 'dog':
        return True, decoded_preds[1]
    else:
        return False, None

# تست تابع برای یک تصویر شامل یک سگ
result, category = predict_dog('dog.jpg')
if result:
    #return True
    print('تشخیص داده شده است که تصویر شامل یک سگ است و دسته‌بندی آن', category, 'است')
else:
    #return False
    print('تشخیص داده شده است که تصویر شامل سگ نیست')