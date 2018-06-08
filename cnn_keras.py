from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import regularizers
from keras import backend as K
import matplotlib.pyplot as plt

# load mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
training batch size 
how many possible outputs (10 available digits)
epochs of learning
"""
batch_size = 10
number_of_classes = 10
epochs = 1
seed = None
mean = 0
stddev = 0.05
eta = 0.03
lambda_l2 = 0.01

#weights
kernel_initializer = keras.initializers.TruncatedNormal(mean = mean,
													 stddev = stddev,
													 seed = seed) 

optimizer = keras.optimizers.SGD(lr = eta, nesterov = True)

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

model.add(Conv2D(filters = 32, 
	kernel_size = (5,5), 
	activation = 'relu', 
	input_shape = input_shape,
	kernel_initializer = kernel_initializer,
	padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters = 40, 
	kernel_size = (5,5), 
	activation = 'relu', 
	input_shape = (1, 12, 12),
	kernel_initializer = kernel_initializer,
	padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, 
	activation='relu',
	kernel_regularizer=regularizers.l2(lambda_l2)))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
				optimizer=optimizer,
				metrics=['accuracy'])

history = model.fit(x_train, y_train, 
	batch_size=batch_size, 
	epochs=epochs, 
	verbose=1, 
	validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss: ", score[0])
print("Accuracy: ", score[1])

# plotting data for accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy(epoch)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# plot the data for loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss(epoch)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print("learning rate: ", eta)
print("batch size: ", batch_size)
print("lambda l2: ", lambda_l2)
print(model.summary())

#save model to json file
json_model = model.to_json()
with open("model.json", "w") as file:
	file.write(json_model)
# save weights
model.save_weights("model.h5")
print("model saved")