from neural_network import *

if __name__ == "__main__":
    # Loading training sample
    train_images, train_labels, test_images, test_labels = load_mnist(
    'data/MNIST/raw/train-images-idx3-ubyte',
    'data/MNIST/raw/train-labels-idx1-ubyte'
    )
