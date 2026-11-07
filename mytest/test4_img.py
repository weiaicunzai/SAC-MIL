import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

datasize = [16, 64, 128, 256]
diff_num = []
for i in range(len(datasize)):
    path = f"mytest/data200_{datasize[i]}.csv"
    df = pd.read_csv(path)
    non_zero_counts = (df != 0).sum()
    diff_num.append([])
    for index, value in non_zero_counts.items():
        diff_num[i].append(value)
# 设置颜色列表
colors = ['#B393FA', '#69FA7F', '#FAB369']
color_labels = ['Cycle-MLP', 'Chord-Mixer', 'SAC-Mixer']  # 定义每种颜色对应的标签
bar_width = 0.5
# 创建画布和子图，改为2x2布局
fig, axs = plt.subplots(2, 2, figsize=(20, 14))  # 调整尺寸以适应2x2布局
# 将axs转换为一维数组以便于索引
axs = axs.ravel()
# 绘制每个子图的柱状图
for i in range(len(diff_num)):
    x = np.arange(len(diff_num[i]))
    max_height = 0
    for j, color in enumerate(colors):
        bar = axs[i].bar(x[j], diff_num[i][j], width=bar_width, color=color, label=color_labels[j])
        # 在每个柱形图上显示数值
        axs[i].text(bar[0].get_x() + bar[0].get_width()/2, bar[0].get_height(),
                    int(bar[0].get_height()), ha='center', va='bottom', fontsize=26)
        height = bar[0].get_height()
        max_height = max(max_height, height)
    # 设置y轴为对数刻度
    axs[i].set_yscale('log')
    # 设置y轴刻度标签的字体大小
    axs[i].tick_params(axis='y', which='both', labelsize=26)  # 调整y轴刻度标签的字体大小
    # 计算并设置y轴的上限，给最大值增加一定的比例作为缓冲区
    axs[i].set_ylim(top=max_height * 2.3)  # 设置y轴范围
    # 设置x轴刻度标签
    axs[i].set_xticklabels([])  # 将所有横坐标标签设为空字符串
    # 设置子图标题
    axs[i].text(0.5, -0.08, f'({chr(97+i)})',
                ha='center', va='top', transform=axs[i].transAxes, fontsize=26)
    # 关闭次要刻度线
    axs[i].minorticks_off()
# 创建统一的图例
# fig.legend(color_labels, prop={'size': 26}, bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
# fig.legend(color_labels, prop={'size': 26}, ncol=len(colors), frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1))
# fig.legend(color_labels, prop={'size': 26}, bbox_to_anchor=(1.05, 0.5), loc='center left', frameon=False)
# 通过调整wspace和hspace值来增加子图间的水平和垂直间距
plt.subplots_adjust(wspace=0.3, hspace=0.2, right=0.85)
# 显示图形
# plt.show()
plt.savefig("mytest/img_bar.pdf")