موضوع : تمرین درس شبکه های عصبی( پردازش تصویر–شناساییالگو سگ)

برایتشخیصسگ در تصویر،میتوانیماز شبکههایعصبیاختصاصی یا سفارشی استفاده کنیم یا از شبکه
های عصبی ازپیشآموزشدیدهمانندVGG،ResNetویاInceptionاستفاده کنیم. در اینجابرایمثال از
شبکهVGG16استفادهشده است.
کلکدVGG 16:
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input,
decode_predictions
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
کد کامل شبکه شفارشی :
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
#بارگیریمدل شبکه عصبیسفارشی
model = tf.keras.models.load_model('my_custom_model.h5')
#تابع تشخیصسگ در تصویر
def dog_detector(image_path):
#بارگیریتصویربا استفاده از کتابخانه keras.preprocessing.image
image = load_img(image_path, target_size=(224, 224))
#تبدیلتصویربه آرایه numpy
image = img_to_array(image)
# تغییرشکلآرایهبه شکلیکه شبکه عصبیمورد انتظار دارد
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# مقیاسدادنتصویر
image = image.astype('float32') / 255.0
# پیشبینیبا استفاده از شبکه عصبیسفارشی
prediction = model.predict(image)
# بررسیآیاتصویرحاویسگ استیاخیر
if prediction[0] >= 0.5:
return True
else:
return False
#تست تابع تشخیصسگ با چند تصویر
image_paths = ['test_dog.jpg', 'test_cat.jpg', 'test_bird.jpg']
for path in image_paths:
if dog_detector(path):
return true
else:
return false
بخشیاز کد که مدل شبکه عصبیرا شامل میشود:
مدل آماده16VGG:
# load the pre-trained VGG16 model
model = VGG16(weights='imagenet')
کد شبکه اختصاصی :
import tensorflow as tf
# تعریفشبکه عصبیسفارشی
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
tf.keras.layers.MaxPooling2D((2,2)),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D((2,2)),
tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D((2,2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])
# کامپایلشبکه عصبی
model.compile(optimizer='adam', loss='binary_crossentropy',
metrics=['accuracy'])
# آموزش شبکه عصبیبا دادههایمورد نظر
model.fit(train_data, train_labels, epochs=10, validation_data=(val_data,
val_labels))
بخشیاز کد که تابع اجراییرا شامل میشود:
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
توضیحیدرباره شبکه عصبیاستفاده شده و کد شبکه عصبیسفارشی:.
درایننمونه، از شبکهVGG16پیشآموزشدیدهبر رویمجموعه دادهImageNetاستفاده شده است. این
شبکه دارای۱۶الیهیپیچیدهایاست که برایدستهبندیتصاویربه کار گرفته میشود.
درکدشبکه سفارشی،یکشبکه عصبیسفارشیبا چند الیهکانولوشنیو الیههایتماما متصل تعریفشده
است. در اینشبکه، ابتدا۳الیهکانولوشنیبا فیلترهای۳x3تعریفشدهاند و سپس برایکاهش ابعاد تصاویر،
از الیههایMaxPoolingاستفاده شده است. در انتها، تصویربه صورتیکبرداریکبعدیتبدیلشده و با
استفاده از الیههایتماما متصل، خروجینهاییمحاسبه شده است.
دربارهکدشبکه سفارشی، بایدتوضیحدهم که در شبکه عصبیسفارشی،برایتشخیصیکدستهبندیخاص،
مانند تشخیصسگ در تصاویر،بایددر الیهآخر از تابع فعالسازیsoftmaxبه جایsigmoidاستفاده کنید
و تعداد نورونهایخروجیرا با توجه به تعداد دستهبندیهاتعیینکنیم. همچنین، بایددادههایآموزشیو
اعتبارسنجیرا به شکلیآماده کنیمکه مطابق با ساختار شبکه عصبیسفارشیماباشند.
توضیحیدرباره کل پروژه:
اینپروژه، به عنوانیکنمونه ساده از تشخیصسگ در تصاویر،شامل ساختیکشبکه عصبی
پیشآموزشدیدهبر رویمجموعه دادهImageNetاست.
توضیحیدرباره خروجی:
خروجیاینپروژه شاملیکعدد بولیناست که نشان میدهدتصویرورودیحاویسگ استیاخیر. برایاین
منظور، تصویرورودیبه تابعclassify_dogداده میشودو خروجیآن بررسیمیشود.
test_dog.jpg contains a dog
test_cat.jpg does not contain a dog
test_bird.jpg does not contain a dog
