from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

file = pd.read_csv("./GroupData.csv")
df = pd.DataFrame(file)

# 获得完整数据的组号
data = []
a_label = []
d_label = []
for index in range(0, 55):
    data.append(df.ix[index][5:])
    if df.ix[index][4] == 1:
        a_label.append([1, 0, 0])
    else:
        if df.ix[index][4] == 2:
            a_label.append([0, 1, 0])
        else:
            a_label.append([0, 0, 1])

    # d_label.append(df.ix[index][3])

x = np.array(data)
a = np.array(a_label)
# d = np.array(d_label)
x_train, x_test, a_train, a_test = train_test_split(x, a, test_size=0.25)

# 第一层网络连接a1_r1到rt_d1_r24 一行共288个数据
inputSize = 288
# 导出分类结果 leader编号
outputSize = 3
hiddenSize = 23
trainTimes = 10000

# 第一层
inputLayer = tf.compat.v1.placeholder(tf.float32, shape=[None, inputSize])
# 隐藏层
hiddenWeight = tf.Variable(tf.truncated_normal([inputSize, hiddenSize], mean=0, stddev=0.01))
hiddenBias = tf.Variable(tf.truncated_normal([hiddenSize]))
hiddenLayer = tf.add(tf.matmul(inputLayer, hiddenWeight), hiddenBias)
hiddenLayer = tf.nn.sigmoid(hiddenLayer)
# 输出层
outputWeight = tf.Variable(tf.truncated_normal([hiddenSize, outputSize], mean=0, stddev=0.01))
outputBias = tf.Variable(tf.truncated_normal([outputSize], mean=0, stddev=0.01))
outputLayer = tf.add(tf.matmul(hiddenLayer, outputWeight), outputBias)
outputLayer = tf.nn.sigmoid(outputLayer)
# 输出层标签和输入层一样采用占位法
outputLabel = tf.compat.v1.placeholder(tf.float32, shape=[None, outputSize])

# 损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=outputLabel, logits=outputLayer))
# 最小化损失函数
optimizer = tf.train.AdadeltaOptimizer()
target = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_dict = {inputLayer: x_train, outputLabel: a_train}

    predictionStep = []
    for i in range(trainTimes):
        sess.run(target, feed_dict=feed_dict)
        if i % 1000 == 0:
            corrected = tf.equal(tf.argmax(outputLabel, 1), tf.argmax(outputLayer, 1))
            accuracy = tf.reduce_mean(tf.cast(corrected, tf.float32))
            accuracyValue = sess.run(accuracy, feed_dict=feed_dict)
            print(i, 'train set accuracy:', accuracyValue)
            # lossVal = sess.run(loss, feed_dict=feed_dict)
            # print("步骤：%d,loss:%f" % (i, lossVal))

    corrected = tf.equal(tf.argmax(outputLabel, 1), tf.argmax(outputLayer, 1))
    accuracy = tf.reduce_mean(tf.cast(corrected, tf.float32))
    accuracyValue = sess.run(accuracy, feed_dict={inputLayer: x_test, outputLabel: a_test})
    print("accuracy on test set:", accuracyValue)

    sess.close()

