import numpy as np
import pickle
import wget
import tarfile
import os
from keras.preprocessing.image import ImageDataGenerator

# 이 파일(datasets.py)을 직접 터미널에서 실행할 경우 실행되는 코드
if __name__ == "__main__":
    # cifar-10 데이터 다운로드
    current_path = os.getcwd()
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    wget.download(url, out = current_path)

    # 압축 풀기
    tar = tarfile.open(current_path + "/cifar-10-python.tar.gz")
    tar.extractall(path = current_path)
    tar.close()

    # 압축 파일 삭제
    os.remove(current_path+"/cifar-10-python.tar.gz")

    # 폴더명 '/data'로 변경
    os.rename(current_path+"/cifar-10-batches-py", current_path+"/data")
    


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding = "bytes")
    return dict

def pickle_to_images_and_labels(root):
    data = unpickle(root)
    data_images = data[b'data'] / 255
    data_images = data_images.reshape(-1, 32, 32, 3).astype("float32")
    data_labels = data[b'labels']
    return data_images, data_labels

def data_loader(batch_size):
    current_path = os.getcwd()
    images1, label1 = pickle_to_images_and_labels(current_path + "/data/data_batch_1")
    images2, label2 = pickle_to_images_and_labels(current_path + "/data/data_batch_2")
    images3, label3 = pickle_to_images_and_labels(current_path + "/data/data_batch_3")
    images4, label4 = pickle_to_images_and_labels(current_path + "/data/data_batch_4")
    images5, label5 = pickle_to_images_and_labels(current_path + "/data/data_batch_5")

    test_images, test_labels = pickle_to_images_and_labels(current_path + "/data/test_batch")

    train_images = np.concatenate([images1, images2, images3, images4, images5], axis = 0)
    train_labels = np.concatenate([label1, label2, label3, label4, label5], axis = 0)
    test_images = np.concatenate([test_images], axis = 0)
    test_labels = np.concatenate([test_labels], axis = 0)

    idg = ImageDataGenerator(horizontal_flip=True) # data augmentation
    train_generator = idg.flow(train_images, y = train_labels, batch_size = batch_size)
    test_generator = idg.flow(test_images, y = test_labels, batch_size = batch_size,
                                            shuffle = False)

    return train_generator, test_generator

