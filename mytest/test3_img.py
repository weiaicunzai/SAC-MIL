import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import gaussian_filter1d
#0：wsi_mixer
#1：cylemlp
#2：chordmixer
#(a)cycle mlp, (b) chord mixer (c)  mlpmixer
# idx = 0
total=200
n=100
path_csv=f"mytest/data{total}_{n}.csv"
path_jpg=f"mytest/img{total}_{n}.pdf"
df = pd.read_csv(path_csv)
# df2,df0,df1 = df.iloc[:,0],df.iloc[:,1],df.iloc[:,2]
# df=df.iloc[:,[1,2,0]]
# df = df.clip(lower=0, upper=10)
# print(df)

# colors=['blue','green','red']
colors=['#B393FA', '#69FA7F', '#FAB369']
color_labels = ['Cycle-MLP', 'Chord-Mixer', 'WSI-Mixer']  # 定义每种颜色对应的标签
line_width=4
alpha_value=0.8
N = 100
df = df.rolling(window=N).mean().dropna()

TSBOARD_SMOOTHING = 0.99 #, 0.85, 0.99
fig, axs = plt.subplots(3, 1, figsize=(15, 20)) # 调整图像尺寸
for j in range(3):  # 对前三列进行操作
    ewm_data = df.iloc[:, j].ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
    # ewm_data=ewm_data/ewm_data.max()
    # axs[j].plot(ewm_data, label=f'Column {j+1}', alpha=alpha_value, color=colors[j])
    axs[j].plot(ewm_data,  alpha=alpha_value, color=colors[j],linewidth=line_width)
    # axs[j].set_title(f'({chr(97+j)}) Smoothing = {TSBOARD_SMOOTHING}')
    # axs[j].set_title(f'({chr(97+j)})', pad=15)
    axs[j].text(0.5, -0.28, f'({chr(97+j)})', ha='center', va='center', transform=axs[j].transAxes, fontsize=36)
    axs[j].grid(alpha=0.3)
    #调整y轴字体
    axs[j].tick_params(axis='y', which='both', labelsize=36)  # 调整y轴刻度标签的字体大小
    axs[j].tick_params(axis='x', which='both', labelsize=36)  # 调整y轴刻度标签的字体大小
# fig.legend(prop={'size': 17})  # 显示图例以区分不同的曲线
fig.legend(color_labels, prop={'size': 36}, ncol=len(colors), frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.96))
# 调整子图之间的垂直距离
plt.subplots_adjust(hspace=0.5)
# plt.tight_layout()
plt.savefig(path_jpg)
# plt.show()
