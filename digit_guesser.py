from keras.models import load_model
import keras
from keras.preprocessing import image
import numpy as np 
import tkinter as tk
import tkinter.messagebox as messagebox
from tkinter import *
import time
from tkinter import filedialog
from keras.models import model_from_json


# image size
rows, columns = 28, 28

root = tk.Tk()
panel = tk.Label(root)
panel.pack()

def load_image():
	### display the image
	imagefile = filedialog.askopenfile(parent = root, mode = "rb", title = "Choose an image!")
	path_to_img = imagefile.name
	display_image = PhotoImage(file = path_to_img)
	display_image = display_image.zoom(12, 12)
	panel.configure(image = display_image)
	panel.image = display_image

	### use the image in keras
	img = image.load_img(path_to_img, target_size = (rows, columns), grayscale = True)
	print(img.size)

	### load the network architecture from json file
	json_file = open("model.json", "r")
	json_model = json_file.read()
	json_file.close()
	model = model_from_json(json_model)
	### load the weights
	model.load_weights("model.h5")
	print("Successfully loaded model") 
	
	# compile the model
	model.compile(loss = "binary_crossentropy",
				 	optimizer = keras.optimizers.SGD(lr = 0.03, nesterov = True),#"rmsprop"
				 	metrics = ["accuracy"])

	# test the image
	test = image.img_to_array(img)
	test = np.expand_dims(test, axis = 0)
	# normalize the image from [0,255] to [0,1] pixel gray level
	test /= 255
	vect = np.vstack([test])


	## PREDICT!!
	classes = model.predict_classes(vect, batch_size = 10)
	print(classes)
	messagebox.showinfo("Guess", "This is a " + str(classes[0]))





b_load = tk.Button(root, text = "Load image and check the digit!", command = load_image)

b_load.pack()

root.geometry("600x400")
root.title("Digit Guesser")
#main loop
while True:
	root.update_idletasks()
	root.update()
	time.sleep(0.01)


