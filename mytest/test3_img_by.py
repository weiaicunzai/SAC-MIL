import math
import csv
import matplotlib.pyplot as plt


def exponential_moving_average(data, alpha=0.1):
    """
    计算指数移动平均值
    :param data: 输入数据列表或数组
    :param alpha: 平滑系数，范围在 [0, 1] 之间
    :return: 平滑后的数据
    """
    smoothed = []
    current_smoothed_value = data[0]

    for value in data:
        current_smoothed_value = alpha * value + (1 - alpha) * current_smoothed_value
        smoothed.append(current_smoothed_value)

    # return np.array(smoothed)
    return smoothed

def normalize(data):
    max_value = max(data)
    data = [v / max_value for v in data]
    return data

def standardize(data):
    """
    标准化数据：将每个值减去均值再除以标准差
    :param data: 输入数据列表
    :return: 标准化后的数据
    """
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std = math.sqrt(variance)

    # 防止标准差为0的情况
    if std == 0:
        raise ValueError("Standard deviation is zero; cannot standardize data.")

    standardized_data = [(x - mean) / std for x in data]
    return standardized_data


# data_a = normalize(data_a)


# 打开一个CSV文件进行读取
# with open('mytest/data200.csv', 'r') as file:
# with open('mytest/data2.csv', 'r') as file:
with open('mytest/data1.csv', 'r') as file:
    reader = csv.reader(file)

    # 读取数据
    data = list(reader)

# N = 100
alpha=0.1
# 将每一列的数据存储为列表
data_a = [float(row[0]) * 100 for row in data]
# max_a = max(data_a)
print('max a', max(data_a))
data_a = exponential_moving_average(data_a, alpha=alpha)
print('max a', max(data_a))
# data_a = [x / max_a for x in data_a]
# data_a = normalize(data_a)
# data_a = standardize(data_a)
print('max a', max(data_a))



data_b = [float(row[1]) * 100 for row in data]
max_b = max(data_b)
data_b = exponential_moving_average(data_b, alpha=alpha)
# data_b = standardize(data_b)
# data_b = [x / max_b for x in data_b]
print(len(data_b))
print('max', max(data_b))

data_c = [float(row[2]) for row in data]
max_c = max(data_c)
data_c = exponential_moving_average(data_c, alpha=alpha)
# data_c = normalize(data_c)
# data_c = [x / max_c for x in data_c]

# print(data_c.max(), data_c.min())

top = 100


# fig, axs = plt.subplots(1, 3)
# fig, axs = plt.subplots(1, 1)
fig, axs = plt.subplots(1, 3, figsize=(50, 5))
print('max', max(data_c))

# 绘制直方图
axs[0].plot(range(len(data_a)), data_a, color='skyblue', linestyle='-')
# axs[0].hist(data_a, bins=100, color='skyblue')
axs[0].set_title('(a)')
# # print('1', max(data_a))
# # print('1', max(data_b))
# # print('1', max(data_c))
# # axs[0].set_ylim(0,10)
axs[0].set_ylim(0, top)

# print(type(data_a))
# print(len(data_a))
import numpy as np
b = np.array(data_b)

print(b.shape)
print((b != 0).sum())
print(b.max(), b.min())
# import sys; sys.exit()
# for i in data_b:
    # if i > 0:
        # print(i)
print(111, len(data_b))
axs[1].plot(range(len(data_b)), data_b, color='lightgreen', linestyle='-')
axs[1].set_title('(b)')
# axs[1].set_ylim(0,2500)
# axs[1].set_ylim(0, 1)
axs[1].set_ylim(0, top)

axs[2].plot(range(len(data_c)), data_c, color='salmon', linestyle='-')
axs[2].set_title('(c)')
# axs[2].set_ylim(0,0.001)
axs[2].set_ylim(0, top)

# 调整布局
plt.tight_layout()
plt.savefig('mytest/test.jpg')
