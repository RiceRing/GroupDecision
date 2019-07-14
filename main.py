from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

file = pd.read_csv("./data.csv")
df = pd.DataFrame(file)

# 获得完整数据的组号
data = []
a_label = []
d_label = []

for index in range(0, 55):
    data.append(df.ix[index][2:])
    # a_label.append(df.ix[index][1])
    if df.ix[index][1] == 1:
        a_label.append(([1], [1, 0, 0]))
    else:
        if df.ix[index][1] == 2:
            a_label.append(([2], [0, 1, 0]))
        else:
            a_label.append(([3], [0, 0, 1]))

data = np.array(data)
a_label = np.array(a_label)
x_train, x_test, a_train, a_test = train_test_split(data, a_label, test_size=0.25)

x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
# a_train = a_train.reshape((a_train.shape[0], 1, a_train.shape[1]))
# a_test = a_test.reshape((a_test.shape[0], 1, a_test.shape[1]))

model = Sequential()
model.add(LSTM(units=20, activation='tanh', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(3, activation='relu'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, a_train, epochs=100, batch_size=20, validation_data=(x_test, a_test), shuffle=False)

score = model.evaluate(x_test, a_test)
print(model.metrics_names)
print('Test score:', score)

# x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))


