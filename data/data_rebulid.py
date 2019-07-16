import pandas as pd


file = pd.read_csv("./RebuildRound.csv")
df = pd.DataFrame(file)
length = len(df.ix[0][0:])
cols = list(df)
# a1_r1 = 5; a2_r1 = 5+24 ; a3_r1 = 5+2*24
minus = 0
for i in range(5, 5+24*6, 6):
    # for j in range(1, 6):
    #     cols.insert(5+j, cols.pop(5+j*24))
    for j in range(1, 3):
        cols.insert(i+j, cols.pop(i+j*(24-minus)))
    count = 0
    for k in range(3, 6):
        print(i+144+count*(24-minus))
        cols.insert(i+k, cols.pop(i+144-minus*3+count*(24-minus)))
        count += 1

    minus += 1
# cols.insert(8, cols.pop(149))
# cols.insert(9, cols.pop(173))
# cols.insert(10, cols.pop(197))
# print(cols.index('rt_a1_r2')) # 152

df = df.loc[:, cols]
# df.to_csv("./AttackRound.csv")

file = pd.read_csv("./RebuildRound2.csv")
df = pd.DataFrame(file)
length = len(df.ix[0][0:])
cols = list(df)
# a1_r1 = 5; a2_r1 = 5+24 ; a3_r1 = 5+24
minus = 0
for i in range(5, 5+24*6, 6):
    # for j in range(1, 6):
    #     cols.insert(5+j, cols.pop(5+j*24))
    for j in range(1, 3):
        cols.insert(i+j, cols.pop(i+j*(24-minus)))
    count = 0
    for k in range(3, 6):
        print(i+144+count*(24-minus))
        cols.insert(i+k, cols.pop(i+72-minus*3+count*(24-minus)))
        count += 1

    minus += 1
# cols.insert(8, cols.pop(149))
# cols.insert(9, cols.pop(173))
# cols.insert(10, cols.pop(197))
# print(cols.index('rt_a1_r2')) # 152

df = df.loc[:, cols]
df.to_csv("./RespondRound.csv")
