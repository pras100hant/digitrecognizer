import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

with open('test.pkl','wb') as f:
   pickle.dump(x_test,f)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
#model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
#model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

#sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 4)



val_loss, val_acc = model.evaluate(x_test, y_test)
print("Loss : ",val_loss, "Acc : ", val_acc)

model.save('epic_num_reader.model')

