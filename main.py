from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file = pd.read_csv("./GroupData.csv")
df = pd.DataFrame(file)

# 获得完整数据的组号
data = []
a_label = []
d_label = []
for index in range(0, 55):
    data.append(df.ix[index][5:])
    a_label.append(df.ix[index][3])
    d_label.append(df.ix[index][2])


# 拆分训练数据和测试数据
x = np.array(data)
a = np.array(a_label)
d = np.array(d_label)
x_train, x_test, y_train, y_test = train_test_split(x, a, test_size=0.2)

clf = neighbors.KNeighborsClassifier(algorithm="kd_tree")
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)
print("attack test：", y_test)
print("attack predict:", y_predict)
score = accuracy_score(y_test, y_predict)
print("Test score:", score)

x_train, x_test, y_train, y_test = train_test_split(x, d, test_size=0.2)

clf = neighbors.KNeighborsClassifier(algorithm="kd_tree")
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)
print("defense test：", y_test)
print("defense predict:", y_predict)
score = accuracy_score(y_test, y_predict)
print("Test score:", score)