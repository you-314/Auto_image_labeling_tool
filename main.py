import numpy as np
import cupy as cp
import pandas as pd
from conv import Convolution as cnv
from pool import Pooling as pl
import layer as ly
import matplotlib.pyplot as plt
import os
import random
from simple_net import SimpleConvNet as scn
from sklearn.model_selection import train_test_split
from trainer import Trainer
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder



# def load_data(DATADIR, CATEGORIES, IMG_SIZE):
#     training_data = []
#     for class_num, category in enumerate(CATEGORIES):
#         path = os.path.join(DATADIR, category)
#         for image_name in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, image_name),)  # 画像読み込み
#                 img_resize_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # 画像のリサイズ
#                 training_data.append([img_resize_array, class_num])  # 画像データ、ラベル情報を追加
#             except Exception as e:
#                 pass
    
#     return training_data

# def create_dataset(DATADIR, CATEGORIES, IMG_SIZE):
#     training_data = load_data(DATADIR, CATEGORIES, IMG_SIZE)
#     random.shuffle(training_data)  # データをシャッフル

#     X_train = []  # 画像データ
#     y_train = []  # ラベル情報

#     # データセット作成
#     for feature, label in training_data:
#         X_train.append(feature)
#         y_train.append(label)

#     # numpy配列に変換
#     X_train = np.array(X_train)
#     y_train = np.array(y_train)

#     # データセットの確認
#     # for i in range(0, 4):
#     #     print("学習データのラベル：", y_train[i])
#     #     plt.subplot(2, 2, i+1)
#     #     plt.axis('off')
#     #     plt.title(label = 'Dog' if y_train[i] == 0 else 'Cat')
#     #     img_array = cv2.cvtColor(X_train[i], cv2.COLOR_BGR2RGB)
#     #     plt.imshow(img_array)
#     return X_train, y_train


# #ミニバッチ作成
# def minibatch(batch_size, data_x, data_y):
#     data_size = data_x.shape[0]
#     batch_i = np.random.choice(data_size, batch_size)
#     mini_x, mini_y = [], []
#     mini_x.append(data_x[batch_i])
#     mini_y.append(data_y[batch_i])
#     return mini_x, mini_y


def print_(x, a):
    print("===================="+a+"======================")
    print(x)
    print("===============================================\n")


def main():
    #画像データロード
    # path = "test_data"
    # category = ["Dog",  "Cat"]
    # x, y = create_dataset(path, category, 100)
    # x = x.transpose(0, 3, 1, 2)
    # print("x="+str(x.shape))
    cifar10 = fetch_openml('CIFAR_10')
    x = cifar10.data.reshape(cifar10.data.shape[0], 3, 32, 32)
    x_train, x_test, t_train, t_test = train_test_split(x, cifar10.target, test_size=0.2, shuffle=True, random_state=0)
   
    x_train_gpu = cp.asarray(x_train)
    x_test_gpu = cp.asarray(x_test)


    enc = OneHotEncoder(categories="auto", sparse=False, dtype=np.float32)
    t_train = enc.fit_transform(t_train.astype(np.int8).reshape(-1, 1))
    t_test = enc.fit_transform(t_test.astype(np.int8).reshape(-1, 1))

    t_train_gpu = cp.asarray(t_train)
    t_test_gpu = cp.asarray(t_test)    

    print_(t_train_gpu, "t_gpu")
    max_epochs = 100
    n, c, h, w = x.shape
    network = scn(input_dim=(c, h, w), 
                conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                hidden_size=100, output_size=10, weight_init_std=0.0008)
                            
    trainer = Trainer(network, x_train_gpu, t_train_gpu, x_test_gpu, t_test_gpu,
                    epochs=max_epochs, mini_batch_size=100,
                    optimizer='SGD', optimizer_param={'lr': 0.001},
                    evaluate_sample_num_per_epoch=10)
    trainer.train()

    # パラメータの保存
    network.save_params("params.pkl")
    print("Saved Network Parameters!")

    # グラフの描画
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


def test():
   #画像データロード
    # path = "test_data"
    # category = ["Dog",  "Cat"]
    # x, y = create_dataset(path, category, 100)
    # x = x.transpose(0, 3, 1, 2)
    # print("x="+str(x.shape))
    cifar10 = fetch_openml('CIFAR_10')
    x = cifar10.data.reshape(cifar10.data.shape[0], 3, 32, 32).astype(np.int32)
    x_train, x_test, t_train, t_test = train_test_split(x, cifar10.target, test_size=0.3, shuffle=True)
    
    x_train_batch = x_train[0:24,:,:,:]
    x_test_batch = x_test[25:35,:,:,:]
    t_train_batch = t_train[0:24]
    t_test_batch = t_test[25:35]
    
    x_train_batch_gpu = cp.asarray(x_train_batch)
    x_test_batch_gpu = cp.asarray(x_test_batch)

    enc = OneHotEncoder(categories="auto", sparse=False, dtype=np.float32)
    t_train_batch = enc.fit_transform(t_train_batch.astype(np.int8))
    t_test_batch = enc.fit_transform(t_test_batch.astype(np.int8))

    t_train_batch_gpu = cp.asarray(t_train_batch)
    t_test_batch_gpu = cp.asarray(t_test_batch)

    max_epochs = 100
    n, c, h, w = x_train_batch.shape
    network = scn(input_dim=(c, h, w), 
                conv_param = {'filter_num': 90, 'filter_size': 5, 'pad': 0, 'stride': 1},
                hidden_size=100, output_size=10, weight_init_std=0.01)
                            
    trainer = Trainer(network, x_train_batch_gpu, t_train_batch_gpu, x_test_batch_gpu, t_test_batch_gpu,
                    epochs=max_epochs, mini_batch_size=1000,
                    optimizer='SGD', optimizer_param={'lr': 0.001},
                    evaluate_sample_num_per_epoch=10)
    trainer.train()

    # パラメータの保存
    network.save_params("params.pkl")
    print("Saved Network Parameters!")

    # グラフの描画
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


if __name__ =="__main__":
    test()
    main()