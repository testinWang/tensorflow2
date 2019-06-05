from __future__ import absolute_import, division,print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

#1 加载数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#2探索数据
print(train_images.shape)

#3 预处理数据
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images/255.0
test_images = test_images/255.0

#4.1.构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation ="softmax")
])

#4.2 编译模型
model.compile(optimizer = "adam",
              loss ="sparse_categorical_crossentropy",
              metrics = ['accuracy'])

#5 训练模型
model.fit(train_images, train_labels, epochs=100)
print("##################model training finish###############")

#6 评估精度
test_loss, test_acc =model.evaluate(test_images, test_labels)
print("Ntest accuracy:", test_acc)

#7 预测
prediction =model.predict(test_images)
print(prediction[0])







