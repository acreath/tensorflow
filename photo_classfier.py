import tensorflow as tf 
from tensorflow import keras

import numpy as np 
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

#将这些数值进行缩放，再输入到神经网络中
train_images = train_images / 255.0
test_images = test_images / 255.0

"""
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()
"""
#build the model
#神经网络的基本建构模块是层

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),#格式化数据，将原有的二维数组切换为一维数组，怎么转换呢
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

#Compile the model 编译模型，损失函数，优化器，监控指标
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Train the model
model.fit(train_images, train_labels, epochs=10)

#Evaluate accuracy
test_loss,test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy: {}'.format(test_acc))

#Make predictions
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])


