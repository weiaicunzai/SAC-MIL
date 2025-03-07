"""
4 16 64
9 122 4096
16 576 10000
25 1842 10000
36 4154 10000
49 6883 10000
64 8905 10000
81 9788 10000
100 9985 10000
121 10000 10000
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 示例数据
x = [i+1 for i in range(10)]  # X轴数据，所有线条共享相同的X轴数据作为例子
y_values = [
    [4, 9, 16, 25, 36, 49, 64, 81, 100, 121],  # 第一条折线的Y轴数据
    [16, 122, 576, 1842, 4154, 6883, 8905, 9788, 9985, 10000],  # 第二条折线的Y轴数据
    [64, 4096, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]   # 第三条折线的Y轴数据
]
# 定义每条线的颜色和对应的图例名称
colors = ['#B393FA', '#69FA7F', '#FAB369']
line_names = ['Cycle-MLP', 'Chord-Mixer', 'WSI-Mixer']

# 创建图表和坐标轴
fig, ax = plt.subplots(figsize=(8, 4))

# 绘制每条折线，并设置颜色、标记和标签
for i, y in enumerate(y_values):
    ax.plot(x, y, color=colors[i], marker='o', linestyle='-', label=line_names[i])
# 添加图例
# ax.legend()
# ax.legend(prop={'size': 12}, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing=1)
ax.legend(prop={'size':15}, loc='best', frameon=False)
# 添加网格线
# ax.grid(True,alpha=0.5)
# 确保所有x轴上的点都被明确地标记
ax.set_xticks(x)  # 设置x轴刻度
ax.set_xticklabels([str(i) for i in x])  # 显示每个点的标签
ax.set_yscale('log')
ax.tick_params(axis='y', which='both', labelsize=15)  # 调整y轴刻度标签的字体大小
ax.tick_params(axis='x', which='both', labelsize=15)  # 调整y轴刻度标签的字体大小
# 关闭次要刻度线
ax.minorticks_off()
# 设置X轴和Y轴标签
# ax.set_xlabel('net layers')
# ax.set_ylabel('diff_num')
# 设置图表标题
# ax.set_title('不同数据系列随时间变化的折线图')
# 去除上边和右边的方框线
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
# 调整布局使标签不被遮挡
plt.tight_layout()
# 显示图表
# plt.show()
plt.savefig("mytest/img_plot.pdf")