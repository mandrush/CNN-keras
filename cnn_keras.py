from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
# load mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
training batch size 
how many possible outputs (10 available digits)
epochs of learning
"""
batch_size = 100
number_of_classes = 10
epochs = 5
#weights
kernel_initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=2)

#each image is 28x28 pixels
rows, columns = 28, 28
print(y_train)
print(x_train.shape[0], x_train.shape[1], x_train.shape[2])

#make the data a 3D arrays for Keras
if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, rows, columns)
	x_test = x_test.reshape(x_test.shape[0], 1, rows, columns)
	input_shape = (1, rows, columns)
else:
	x_train = x_train.reshape(x_train.shape[0], rows, columns, 1)
	x_test = x_test.reshape(x_test.shape[0], rows, columns, 1)
	input_shape = (rows, columns, 1)

#normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

#we need to categorize the y, e.g. for it's 1 for [0,1,0,0,0,0,0,0,0,0]
y_train = keras.utils.to_categorical(y_train, number_of_classes)
y_test = keras.utils.to_categorical(y_test, number_of_classes)

model = Sequential()

#in -> conv2d (32) -> conv2d(32) -> maxpool2d(2) -> dropout -> conv2d(64) -> conv2d(64) -> maxpool2d(2) -> dropout
# -> flatten (just to convert to 1D) -> two classifiers -> output
model.add(Conv2D(filters = 32, 
	kernel_size = (5,5), 
	activation = 'relu', 
	input_shape = input_shape,
	padding = 'same'))
model.add(Conv2D(filters = 32, 
	kernel_size = (5,5), 
	activation = 'relu', 
	input_shape = input_shape,
	padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#commenting out the second set of layers because it was too much for my pc lol
"""model.add(Conv2D(filters = 64, 
	kernel_size = (5,5), 
	activation = 'relu', 
	input_shape = input_shape,
	padding = 'same'))
model.add(Conv2D(filters = 64, 
	kernel_size = (5,5), 
	activation = 'relu', 
	input_shape = input_shape,
	padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))"""

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
				optimizer=keras.optimizers.Adadelta(),
				metrics=['accuracy'])

model.fit(x_train, y_train, 
	batch_size=batch_size, 
	epochs=epochs, 
	verbose=1, 
	validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss: ", score[0])
print("Accuracy: ", score[1])

