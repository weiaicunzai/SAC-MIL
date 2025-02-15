import csv
import matplotlib.pyplot as plt

# 打开一个CSV文件进行读取
with open('mytest/data.csv', 'r') as file:
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
axs[0].bar(range(len(data_a)), data_a, color='skyblue')
axs[0].set_title('(a)', loc='bottom')
axs[0].set_ylim(0,25)

axs[1].bar(range(len(data_b)), data_b, color='lightgreen')
axs[1].set_title('(b)', loc='bottom')
axs[1].set_ylim(0,1200)

axs[2].bar(range(len(data_c)), data_c, color='salmon')
axs[2].set_title('(c)', loc='bottom')
axs[2].set_ylim(0,0.001)

# 调整布局
plt.tight_layout()
plt.show()
