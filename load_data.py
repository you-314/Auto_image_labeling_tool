import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np

class Load_data:
    def __init__(self, DATADIR, CATEGORIES, IMG_SIZE):
        self.DATADIR = DATADIR
        self.CATEGORIES = CATEGORIES
        self.IMAGES = IMG_SIZE
        self.training_data = np.array([])

        for class_num, category in enumerate(self.CATEGORIES):
            path = os.path.join(self.DATADIR, category)
            for image_name in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, image_name),)  # 画像読み込み
                    img_resize_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))  # 画像のリサイズ
                    self.training_data.append([img_resize_array, class_num])  # 画像データ、ラベル情報を追加
                except Exception as e:
                    pass

    def create_training_data(self):
        random.shuffle(self.training_data)  # データをシャッフル

        X_train = []  # 画像データ
        y_train = []  # ラベル情報

        # データセット作成
        for feature, label in self.training_data:
            X_train.append(feature)
            y_train.append(label)

        # numpy配列に変換
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        print(X_train.shape)

        return X_train, y_train