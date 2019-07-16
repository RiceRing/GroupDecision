from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
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
    train, test, y_train, y_test = train_test_split(d, label, test_size=0.25)
    train = train.reshape((train.shape[0], 1, train.shape[1]))
    test = test.reshape((test.shape[0], 1, test.shape[1]))
    y_train = to_categorical(y_train, 3)
    y_test = to_categorical(temp, 3)
    # a_train = a_train.reshape((a_train.shape[0], 1, a_train.shape[1]
    # a_test = a_test.reshape((a_test.shape[0], 1, a_test.shape[1]))
    return train, test, y_train, y_test, temp


def set_model(train, test, y_train, y_test):
    model = Sequential()
    model.add(LSTM(units=16, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train, y_train, epochs=80, batch_size=16, validation_data=(test, y_test), verbose=None, shuffle=False)

    score = model.evaluate(test, y_test)
    predict = []
    for p_index in range(0, len(test)):
        single = model.predict(x_test)[p_index].tolist()
        max_index = single.index(max(single))
        min_index = single.index(min(single))
        mid_index = 3-max_index-min_index
        predict.append(mid_index+1)

    print("predict:", model.predict_classes(test)+1)
    print('Test loss:', score[0], "; Test accuracy:", score[1])
    print(model.predict(x_test))
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
x_train, x_test, a_train, a_test, temp = split(data, a_label)
t = []
for i in temp:
    t.append(i+1)
print("test:", t)
a_model = set_model(x_train, x_test, a_train, a_test)
predict_a = expand(predict_a)
result_a = a_model.predict_classes(predict_a)

x_train, x_test, d_train, d_test, temp = split(data, d_label)
t = []
for i in temp:
    t.append(i+1)
print("test:", t)
d_model = set_model(x_train, x_test, d_train, d_test)
predict_d = expand(predict_d)
result_d = d_model.predict_classes(predict_d)
print("attack leader:", result_a+1)
print("defense leader:", result_d+1)
