import gzip
import numpy as np

def load_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

def load_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    images = images.reshape(-1, 1, 28, 28)
    return images

def load_fashion_mnist():
    x_train = load_images('dataset/fashion/train-images-idx3-ubyte.gz')
    t_train = load_labels('dataset/fashion/train-labels-idx1-ubyte.gz')

    x_test = load_images('dataset/fashion/t10k-images-idx3-ubyte.gz')
    t_test = load_labels('dataset/fashion/t10k-labels-idx1-ubyte.gz')

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return (x_train, t_train), (x_test, t_test)