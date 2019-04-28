import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2


with open('test.pkl','rb') as f:
	x_test = pickle.load(f)

trained_model = tf.keras.models.load_model('epic_num_reader.model')

while(1):
	x_test_index = int(input("Enter the index of image in test dataset : "))
	predictions = trained_model.predict(x_test)
	print(np.argmax(predictions[x_test_index]))
	##plt.imshow(x_test[x_test_index])
	plt.imshow(x_test[x_test_index], cmap = plt.cm.binary)
	plt.show()
