
# coding: utf-8

# In[5]:


# coding: utf-8
import sys, os
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from PIL import Image
import numpy as np

image_files = ["food/Chicken/chicken_01.jpg",
               "food/Chicken/chicken_02.jpg",
               "food/Kimchi/kimchi15.jpg",
               "food/Kimchi/kimchi07.jpg",
               "food/Samgyeobsal/Samgyeobsal04.jpg"]
image_size = 64
nb_classes = len(image_files)
categories = ["Chicken", "Dolsotbab", "Jeyugbokk-eum", "Kimchi", "Samgyeobsal", "SoybeanPasteStew"]


X = []
files = []
for fname in image_files:
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    in_data = np.asarray(img)
    in_data = in_data.astype("float") / 256
    X.append(in_data)
    files.append(fname)
    
X = np.array(X)



from keras.models import load_model
 
# 모델 파일 읽어오기
model = load_model('my_kfood.h5')


## 예측 실행
pre = model.predict(X)


for i, p in enumerate(pre):
    y = p.argmax()
    print("입력:", files[i])
    print("예측:", categories[i])

