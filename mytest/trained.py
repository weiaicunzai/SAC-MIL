import sys
import os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt


import torch
import torch.nn.functional as F

# from dataset.wsi_dataset import WSIDataset


# from models.mixer import MlpMixer

# dataset = WSIDataset('/data//smb/syh/WSI_cls/', fold=0, dataset='cam16', state='test')



# model = MlpMixer(input_dim=1024, hidden_dim=512, scale=512)

# new_state_dict = {k.replace('model.', ''): v for k, v in torch.load('epoch=130-val_acc=0.8730.ckpt')['state_dict'].items()}
# model.load_state_dict(new_state_dict)

# for i in dataset:
#     print(i)

path = '/data/smb/syh/WSI_cls/cam16/draw/mlp_mixer'

def get_diff(path):

    res = []
    for i in os.listdir(path):
        if '.pt' not in i:
            continue

        if 'zero_center' in i:
            continue

        before = torch.load(os.path.join(path, i))
        print(before.shape)
        i = i.replace('.pt', '_zero_center.pt')

        print(i)
        after = torch.load(os.path.join(path, i))

        diff = after - before
        # print(diff.shape)
        diff = torch.norm(diff, dim=-1, p=1, keepdim=True)
        print(diff.max(), diff.min())
        res.append(diff)

    return res

def get_max_len(diffs):
    max_len = 0
    for diff in diffs:
        length = diff.shape[1]
        if max_len < length:
            max_len = length
    return max_len

def interpolate(diffs):
    max_length = get_max_len(diffs)
    res = []
    print(max_length)
    for diff in diffs:
    # print(f"Original tensor shape: {tensor.shape}")

    # 目标尺寸
        # print(diff.shape)
        # diff = diff.permute(0, 2, 1)
        # print(diff.shape)

        target_size = max_length

        diff = diff.squeeze(-1).unsqueeze(0)  # 转换为 [1, 1, N]
        print(diff.max(), diff.min())
        diff = F.interpolate(diff, size=(target_size,), mode='linear', align_corners=True)
        diff = diff.squeeze(0).unsqueeze(-1)  # 转换回 [1, target_size, 1]
        print(11, diff.max(), diff.min())
        # 使用 interpolate 进行线性插值
        # diff = F.interpolate(diff.unsqueeze(-1), size=(target_size,), mode='linear', align_corners=True).squeeze(-1)
        # diff = diff.permute(0, 2, 1)

        res.append(diff)

    return res

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

def standardize(data):
    import math
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


diffs = get_diff(path)
diffs = interpolate(diffs)
total = 0
for diff in diffs:
    total += diff
    break
    # print(diff.shape)

# total = total / len(diffs)

# total = interpolate(total)
print(total.mean(), total.std(), total.max(), total.min())



total = total.view(-1).tolist()


# total = exponential_moving_average(total, alpha=0.2)
# print(max(total))

total = standardize(total)
total = torch.tensor(total)
print(total.mean(), total.std(), total.max(), total.min())

total = total.tolist()
fig, axs = plt.subplots(1, 3, figsize=(50, 5))
#print('max', max(data_c))
#
## 绘制直方图
total = exponential_moving_average(total, alpha=0.2)
axs[0].plot(range(len(total)), total, color='skyblue', linestyle='-')
#axs[0].set_title('(a)')
## # print('1', max(data_a))
## # print('1', max(data_b))
## # print('1', max(data_c))
## # axs[0].set_ylim(0,10)
#axs[0].set_ylim(0, top)
#
## print(type(data_a))
## print(len(data_a))
#import numpy as np
#b = np.array(data_b)
#
#print(b.shape)
#print((b != 0).sum())
#print(b.max(), b.min())
## import sys; sys.exit()
## for i in data_b:
#    # if i > 0:
#        # print(i)
#print(111, len(data_b))
#axs[1].plot(range(len(data_b)), data_b, color='lightgreen', linestyle='-')
#axs[1].set_title('(b)')
## axs[1].set_ylim(0,2500)
## axs[1].set_ylim(0, 1)
#axs[1].set_ylim(0, top)
#
#axs[2].plot(range(len(data_c)), data_c, color='salmon', linestyle='-')
#axs[2].set_title('(c)')
## axs[2].set_ylim(0,0.001)
#axs[2].set_ylim(0, top)
#
## 调整布局
plt.tight_layout()
plt.savefig('mytest/test.jpg')
#