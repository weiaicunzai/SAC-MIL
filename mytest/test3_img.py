import csv
import matplotlib.pyplot as plt

# 打开一个CSV文件进行读取
with open('mytest/data200.csv', 'r') as file:
    reader = csv.reader(file)

    # 读取数据
    data = list(reader)

# 将每一列的数据存储为列表
data_a = [float(row[1]) for row in data]
data_b = [float(row[2]) for row in data]
data_c = [float(row[3]) for row in data]


fig, axs = plt.subplots(1, 3)
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 绘制直方图
# axs[0].bar(range(len(data_a)), data_a, color='skyblue')
# axs[0].set_title('(a)')
# axs[0].set_ylim(0,25)

# print(type(data_a))
# print(len(data_a))
import numpy as np
b = np.array(data_b)

print(b.shape)
print((b != 0).sum())
print(b.max(), b.min())
# import sys; sys.exit()
axs[1].bar(range(len(data_b)), [1500] * len(data_b), color='lightgreen')
axs[1].set_title('(b)')
axs[1].set_ylim(0,2500)

# axs[2].bar(range(len(data_c)), data_c, color='salmon')
# axs[2].set_title('(c)')
# axs[2].set_ylim(0,0.001)

# 调整布局
plt.tight_layout()
plt.savefig('test.jpg')
