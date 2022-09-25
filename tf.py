# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import gzip

def load_data():
    paths = [
        "train-labels-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ]

    with gzip.open(paths[0], "rb") as lbpath:
        lbpath.read(4) # magic number, ignored, should be 0x00000801
        size = int.from_bytes(lbpath.read(4), 'big')

        y_train = np.frombuffer(lbpath.read(), np.uint8)

    with gzip.open(paths[1], "rb") as imgpath:
        imgpath.read(4) # magic number, ignored, should be 0x00000803
        size = int.from_bytes(imgpath.read(4), 'big')
        rows = int.from_bytes(imgpath.read(4), 'big')
        cols = int.from_bytes(imgpath.read(4), 'big')

        x_train = np.frombuffer(imgpath.read(), np.uint8).reshape(
            size, rows, cols
        )

    with gzip.open(paths[2], "rb") as lbpath:
        lbpath.read(4) # magic number, ignored, should be 0x00000801
        size = int.from_bytes(lbpath.read(4), 'big')

        y_test = np.frombuffer(lbpath.read(), np.uint8)

    with gzip.open(paths[3], "rb") as imgpath:
        imgpath.read(4) # magic number, ignored, should be 0x00000803
        size = int.from_bytes(imgpath.read(4), 'big')
        rows = int.from_bytes(imgpath.read(4), 'big')
        cols = int.from_bytes(imgpath.read(4), 'big')

        x_test = np.frombuffer(imgpath.read(), np.uint8).reshape(
            size, rows, cols
        )

    return (x_train, y_train), (x_test, y_test)

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

def train_model(model, dataset, labels):
    # train
    model.fit(dataset, labels, epochs=10)

def test_model(model, dataset, labels):
    # test
    test_loss, test_acc = model.evaluate(dataset,  labels, verbose=2)

    print('\nTest accuracy:', test_acc)

def main():
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # display first 100 images
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(train_labels[i])
    plt.show()

    # normalize images colour to (0..1)
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = create_model()
    train_model(model, train_images, train_labels)
    test_model(model, test_images, test_labels)
    model.save("model")

if __name__ == "__main__":
    main()
