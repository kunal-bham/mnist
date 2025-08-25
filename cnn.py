from neural_network import *


def cnn_training_loop():
    # 32 and 64 3×3 kernels 
    K1 = np.random.randn(32, 3, 3)
    K2 = np.random.randn(64, 3, 3)

    filters = (K1, K2)
    for i in range(0, 10):
        filters = back_propagation(filters)


if __name__ == "__main__":
    # Loading training sample
    train_images, train_labels, test_images, test_labels = load_mnist(
    'data/MNIST/raw/train-images-idx3-ubyte',
    'data/MNIST/raw/train-labels-idx1-ubyte'
    )

    cnn_training_loop()