# coding: utf-8
# sample
#   > python food_dataset.py ./food/kfood_data.npy ./food/ "Chicken:Dolsotbab:Jeyugbokk-eum:Kimchi:Samgyeobsal:SoybeanPasteStew"

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from PIL import Image
import os, glob
import numpy as np


out_filename = sys.argv[1]  # output filename
root_dir = sys.argv[2]  # image root dir
category = sys.argv[3]  # category name

## 분류 데이터 로딩
root_dir = "./kfood/"
categories = category.split(":")
nb_classes = len(categories)
image_size = 64


## 데이터 변수
X = [] # 이미지 데이터
Y = [] # 레이블 데이터

for idx, cat in enumerate(categories):
    image_dir = root_dir + cat
    files = glob.glob(image_dir + "/" + "*.jpg")
    print(image_dir + "/" +"*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_size, image_size))
        data = np.asarray(img)
        X.append(data)
        Y.append(idx)
        
X = np.array(X)
Y = np.array(Y)


## 학습 데이터 구성하기
X_train, X_test, Y_train, Y_test =     train_test_split(X, Y)
    
xy = (X_train, X_test, Y_train, Y_test)
np.save(out_filename, xy)
