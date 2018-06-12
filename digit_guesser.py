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
from scipy.misc import imread, imresize
import tensorflow as tf
import base64


# image size
rows, columns = 28, 28
global encoded, img


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
	with open(path_to_img, "rb") as png_image:
		encoded = base64.b64encode(png_image.read())

	with open(path_to_img, "wb") as png_image:
		png_image.write(base64.b64decode(encoded))

	print(encoded)
	img = imread(path_to_img, mode = "L")

	### load the network architecture from json file
	json_file = open("model_json", "r")
	json_model = json_file.read()
	json_file.close()
	model = model_from_json(json_model)
	### load the weights
	model.load_weights("model.h5")
	print("Successfully loaded model") 
	graph = tf.get_default_graph()
	
	# compile the model
	model.compile(loss = "categorical_crossentropy",
				 	optimizer = "adam",
				 	metrics = ["accuracy"])

	# make the image suitable for the model
	img = imresize(img, (rows, columns))
	img = img.reshape(1, 28, 28, 1)
	with graph.as_default():
		out = model.predict(img)
		print(out)
		print(np.argmax(out, axis = 1))
		guess = np.argmax(out, axis = 1)

	messagebox.showinfo("Guess", "This is a " + str(guess[0]))





b_load = tk.Button(root, text = "Load image and check the digit!", command = load_image)

b_load.pack()

root.geometry("600x400")
root.title("Digit Guesser")
#main loop
while True:
	root.update_idletasks()
	root.update()
	time.sleep(0.01)


