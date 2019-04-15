from __future__ import (absolute_import, division, print_function,
    unicode_literals)

import numpy
import tensorflow


batch_size=128
hidden_units=256
dropout=0.45

mnist = tensorflow.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
input_size = image_size * image_size

x_train, x_test = x_train / 255.0, x_test / 255.0
                        
num_labels = len(numpy.unique(y_train))

y_train = tensorflow.one_hot(y_train, num_labels)
y_test = tensorflow.one_hot(y_test, num_labels)

model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Flatten(input_shape=(image_size, image_size)),
    tensorflow.keras.layers.Dense(hidden_units, activation='relu'),
    tensorflow.keras.layers.Dropout(dropout),
    tensorflow.keras.layers.Dense(hidden_units, activation='relu'),
    tensorflow.keras.layers.Dropout(dropout),
    tensorflow.keras.layers.Dense(num_labels, activation='softmax')
])

model.summary()

tensorflow.keras.utils.plot_model(model, to_file='plot/mlp-mnist.png', show_shapes=True)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc) )
