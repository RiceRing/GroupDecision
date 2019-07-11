from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file = pd.read_csv("./GroupData.csv")
df = pd.DataFrame(file)

# 获得完整数据的组号
valid_num = []
for index in range(0, 55):
    valid_num.append(df.ix[index][0])

data = []
label = []
file = pd.read_csv("./data/AttackDecision.csv")
df = pd.DataFrame(file)
for index in range(0,85):
    if df.ix[index][0] in valid_num:
        label.append(df.ix[index][1])
        data.append(df.ix[index][2:-1])

# 拆分训练数据和测试数据
x = np.array(data)
y = np.array(label)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf = neighbors.KNeighborsClassifier(algorithm="kd_tree")
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)
print(y_test)
print(y_predict)
score = accuracy_score(y_test, y_predict)
print(score)

# 预测defence
data = []
label = []
file = pd.read_csv("./data/DefenseDecision.csv")
df = pd.DataFrame(file)
for index in range(0,85):
    if df.ix[index][0] in valid_num:
        label.append(df.ix[index][1])
        data.append(df.ix[index][2:-1])

# 拆分训练数据和测试数据
x = np.array(data)
y = np.array(label)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf = neighbors.KNeighborsClassifier(algorithm="kd_tree")
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)
print(y_test)
print(y_predict)
score = accuracy_score(y_test, y_predict)
print(score)
