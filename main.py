from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def expand(da):
    da = np.array(da)
    return np.expand_dims(da, axis=1)


def split(d, l):
    d = np.array(d)
    label = np.array(l)
    label = to_categorical(label, 3)
    x_train, x_test, a_train, a_test = train_test_split(d, label, test_size=0.1)
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    # a_train = a_train.reshape((a_train.shape[0], 1, a_train.shape[1]))
    # a_test = a_test.reshape((a_test.shape[0], 1, a_test.shape[1]))
    return x_train, x_test, a_train, a_test


def set_model(x_train, x_test, a_train, a_test):
    model = Sequential()
    model.add(LSTM(units=16, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, a_train, epochs=80, batch_size=16, validation_data=(x_test, a_test), verbose=2, shuffle=False)

    score = model.evaluate(x_test, a_test)
    print(model.metrics_names)
    print('Test score:', score)
    return model

# x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))


file = pd.read_csv("./GroupData.csv")
df = pd.DataFrame(file)

# 获得完整数据的组号
data = []
a_label = []
d_label = []
predict_a = []
predict_d = []
for index in range(0, 55):
    data.append(df.ix[index][4:])
    a_label.append(df.ix[index][3]-1)
    d_label.append(df.ix[index][2]-1)
for j in range(76, 85):
    predict_a.append(df.ix[j][4:])
    predict_d.append(df.ix[j][4:])
x_train, x_test, a_train, a_test = split(data, a_label)
a_model = set_model(x_train, x_test, a_train, a_test)
predict_a = expand(predict_a)
result_a = a_model.predict_classes(predict_a)

x_train, x_test, d_train, d_test = split(data, d_label)
d_model = set_model(x_train, x_test, d_train, d_test)
predict_d = expand(predict_d)
result_d = d_model.predict_classes(predict_d)
print("attack leader:", result_a+1)
print("defense leader:", result_d+1)
