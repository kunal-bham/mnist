from torchvision import datasets, transforms
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(image_path, label_path):
    # Load images
    with open(image_path, 'rb') as imgf:
        _, num_images, rows, cols = struct.unpack(">IIII", imgf.read(16))
        images = np.frombuffer(imgf.read(), dtype=np.uint8).reshape(num_images, rows, cols)

    # Load labels
    with open(label_path, 'rb') as labf:
        _, num_labels = struct.unpack(">II", labf.read(8))
        labels = np.frombuffer(labf.read(), dtype=np.uint8)

    # image is list of lists (28x28); label is singular number
    images = [image.flatten() for image in images]
    labels = [[1 if i == label else 0 for i in range(10)] for label in labels]

    split_index = int(0.9 * len(images))

    test_images, test_labels = images[split_index:], labels[split_index:]
    train_images, train_labels = images[:split_index], labels[:split_index]

    return train_images, train_labels, test_images, test_labels

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def feed_forward(input_vector, weights, biases):
    W1, W2 = weights
    B1, B2 = biases
    z1 = np.matmul(W1, input_vector) + B1
    a1 = sigmoid(z1)
    z2 = np.matmul(W2, a1) + B2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def cost_function(actual_vector, expected_vector):
    cost = 0
    for actual_elem, expected_elem in zip(actual_vector, expected_vector):
        cost += ((actual_elem - expected_elem)**2)
    return cost

def backprop_calculus_biases(biases, output_layers):
    LEARNING_RATE = 0
    return biases

def backprop_calculus_weights(weights, output_layers):
    LEARNING_RATE = 0
    return weights

def back_propagation(weights, biases):
    total_cost = 0
    batched_output = []
    for train_image, train_label in zip(train_images, train_labels):
            total_cost = 0
            output_layers = feed_forward(train_image, weights, biases)
            total_cost += cost_function(output_layers[3], train_label)
            batched_output.append(output_layers)
            if len(batched_output) == 64:
                weights = backprop_calculus_weights(weights, output_layers)
                biases = backprop_calculus_biases(biases, output_layers)
                batched_output = []
    total_cost /= len(train_images)
    return weights, biases

def training_loop():
    # Randomize weights and biases -1 to 1
    W1 = np.random.rand(128,784)
    B1 = np.random.rand(128,) * 2 - 1

    W2 = np.random.rand(10,128)
    B2 = np.random.rand(10,) * 2 - 1

    weights = (W1, W2)
    biases = (B1, B2)

    for i in range(0, 10):
        weights, biases = back_propagation(weights, biases)

if __name__ == "__main__":
    # Loading training sample
    train_images, train_labels, test_images, test_labels = load_mnist(
    'data/MNIST/raw/train-images-idx3-ubyte',
    'data/MNIST/raw/train-labels-idx1-ubyte'
    )
    
    training_loop()