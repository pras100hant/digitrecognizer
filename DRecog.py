import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2

trained_model = tf.keras.models.load_model('epic_num_reader.model')

def recognize(x):
	img = cv2.imread("digit_images/" + x + ".jpg", 0)
	resized_image = cv2.resize(img,(28,28))
	for i in range(28):
		for j in range(28):
			px_val = resized_image[i,j]
			if px_val>10 and px_val<90:
				px_val += 90
			else:
				px_val =0
			resized_image[i,j] = px_val
		
	resized_images = resized_image[np.newaxis, : , : ]
	##resized_images = tf.keras.utils.normalize(resized_images, axis = 1)

	predictions = trained_model.predict(resized_images)
	##predictions = model.predict(resized_images)
	return (np.argmax(predictions[0]))
	##plt.imshow(resized_image, cmap = plt.cm.binary)
	##plt.show()

