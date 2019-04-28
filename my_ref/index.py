import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2



##mnist = tf.keras.datasets.mnist
##(x_train, y_train), (x_test,y_test) = mnist.load_data()
##x_train = tf.keras.utils.normalize(x_train, axis = 1)
##x_test = tf.keras.utils.normalize(x_test, axis = 1)
##
##with open('test.pkl','wb') as f:
##   pickle.dump(x_test,f)
##
##model = tf.keras.models.Sequential()
##model.add(tf.keras.layers.Flatten())
##model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
###model.add(tf.keras.layers.Dropout(0.5))
##model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
###model.add(tf.keras.layers.Dropout(0.5))
##model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))
##
###sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
##model.compile(optimizer = 'adam',
##             loss = 'sparse_categorical_crossentropy',
##             metrics = ['accuracy'])
##model.fit(x_train, y_train, epochs = 4)
##
##
##
##val_loss, val_acc = model.evaluate(x_test, y_test)
##print("Loss : ",val_loss, "Acc : ", val_acc)
##
##model.save('epic_num_reader.model')



##with open('test.pkl','rb') as f:
##	x_test = pickle.load(f)

new_model = tf.keras.models.load_model('epic_num_reader.model')

##
####while(1):
####	x_test_index = int(input("Enter the index of image in test dataset : "))
####	predictions = new_model.predict(x_test)
####	print(np.argmax(predictions[x_test_index]))
####	##plt.imshow(x_test[x_test_index])
####	plt.imshow(x_test[x_test_index], cmap = plt.cm.binary)
####	plt.show()
##
##
img = cv2.imread("dark_five.jpg",0)
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

predictions = new_model.predict(resized_images)
##predictions = model.predict(resized_images)
print(np.argmax(predictions[0]))
plt.imshow(resized_image, cmap = plt.cm.binary)
plt.show()

