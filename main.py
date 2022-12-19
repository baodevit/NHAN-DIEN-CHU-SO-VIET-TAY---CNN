from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from keras.models import load_model
import numpy as np


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    return x_train, y_train, x_test, y_test


def processing_data():
    x_train, y_train, x_test, y_test = load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def model_cnn(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(units=10))
    model.add(Activation("softmax"))

    model.summary()

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    return model


def training():
    input_shape = (28, 28, 1)
    num_epochs = 10
    num_batch_size = 32
    x_train, y_train, x_test, y_test = processing_data()

    model = model_cnn(input_shape=input_shape)
    model.fit(x=x_train,
              y=y_train,
              epochs=num_epochs,
              batch_size=num_batch_size,
              validation_steps=0.15)
    model.save("model.h5")


def predict(number_pic):
    model = load_model("model.h5")
    x_train, y_train, x_test, y_test = processing_data()

    image = x_test[number_pic]

    img = x_test[number_pic].reshape(28, 28)
    img_reshape = img.reshape(1, 28, 28, 1)

    y_predict = model.predict(img_reshape)
    print("Number: {}".format(np.argmax(y_predict)))
    plt_image(image)


def plt_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    num_pic = int(input("Enter number: "))

    while num_pic > -1:
        predict(number_pic=num_pic)
        num_pic = int(input("Enter number: "))
        if num_pic == -1:
            break
