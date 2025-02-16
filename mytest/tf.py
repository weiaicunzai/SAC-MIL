import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import gaussian_filter1d

idx = 0
df = pd.read_csv("mytest/data1.csv")
df = df.iloc[:, idx]
# df = df.clip(lower=0, upper=10)
print(df)
N = 100
df = df.rolling(window=N).mean().dropna()
print(df)
# smoothed_data = gaussian_filter1d(df, 2)

TSBOARD_SMOOTHING = [0.5, 0.85, 0.99]

smooth = []
for ts_factor in TSBOARD_SMOOTHING:
    smooth.append(df.ewm(alpha=(1 - ts_factor)).mean())

fig, axs = plt.subplots(1, 3, figsize=(50, 5))

for ptx in range(3):
    plt.subplot(1,3,ptx+1)
    # plt.plot(df["value"], alpha=0.4)
    # plt.plot(df.iloc[:], alpha=0.4)
    # plt.bar(df.shape[0], df.iloc[:], alpha=0.3, color='gray')
    plt.plot(df.iloc[:], alpha=0.3, color='gray')
    # plt.plot(smooth[ptx]["value"])
    # plt.plot(smooth[ptx].iloc[:])
    plt.title("Tensorboard Smoothing = {}".format(TSBOARD_SMOOTHING[ptx]))
    plt.grid(alpha=0.3)

plt.savefig('mytest/tf{}.jpg'.format(idx))