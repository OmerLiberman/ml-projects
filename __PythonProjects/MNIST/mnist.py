"""
mnist.py
----------------
Output trained model
"""

import argparse
from numpy import load
import tensorflow as tf


parser = argparse.ArgumentParser(description='Running MNIST Algorithm')
parser.add_argument('--epochs', type=int, help='number of epochs to run', default=12)
parser.add_argument('--batch_size', type=int, help='iteration batch size', default=128)
parser.add_argument('--x_train', help='X_train file. Suppose to be .npy', default='X_train.npy')
parser.add_argument('--y_train', help='y_train file. Suppose to be .npy', default='y_train.npy')
parser.add_argument('--x_test', help='X_test file. Suppose to be .npy', default='X_test.npy')
parser.add_argument('--y_test', help='y_test file. Suppose to be .npy', default='y_test.npy')
args = parser.parse_args()

CLASSES = 10
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

print('cnvrg_tag_batch_size:', BATCH_SIZE)
print('cnvrg_tag_num_classes:', CLASSES)
print('cnvrg_tag_epochs:', EPOCHS)


def init_model(shape):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(shape[0], shape[1], 1)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(CLASSES, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])

    return model


# -----------------------
X_train = load(args.x_train)
y_train = load(args.y_train)

model = init_model((28, 28))

hist = model.fit(
                    X_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS
                )

loss_train, acc_train = hist.history['loss'][0], hist.history['acc'][0]

print('cnvrg_tag_loss_train:', loss_train)
print('cnvrg_tag_accuracy_train:', acc_train)


# -----------------------
X_test = load(args.x_test)
y_test = load(args.y_test)

score = model.evaluate(X_test, y_test)
loss_test, acc_test = score[0], score[1]

print('cnvrg_tag_loss_test:', loss_test)
print('cnvrg_tag_accuracy_test:', acc_test)

model.save('mnist_model.h5')