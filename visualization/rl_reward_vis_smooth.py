import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage.filters import gaussian_filter1d
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default=None)
parser.add_argument('--smoothing', type=float, default=1.0)
parser.add_argument('--begin', type=int, default=0)
parser.add_argument('--end', type=int, default=1000)
args = parser.parse_args()

assert(args.dir!=None)
colors = ['red','blue','green','grey','oend','pink','black']

filenames = os.listdir(args.dir)
if len(filenames)>7:
    print("At Most Can visualize 7 different Frame")
    exit(1)

means = []
stds = []
for fname in filenames:
    arr = np.load(args.dir+'/'+fname)
    min = arr.mean(1).argmin()
    one_cold = np.ones(arr.shape[0])
    one_cold[min] = 0
    arr = arr[one_cold.astype(np.bool)]
    means.append(gaussian_filter1d(arr.mean(0),sigma=args.smoothing))
    stds.append(gaussian_filter1d(arr.std(0),sigma=args.smoothing))
    
for i in range(len(means)):
    x = np.arange(means[i].shape[0])
    plot = plt.plot(x[args.begin:args.end],means[i][args.begin:args.end],color=colors[i],label = filenames[i][:-4])
    plt.fill_between(x[args.begin:args.end],means[i][args.begin:args.end]+stds[i][args.begin:args.end],means[i][args.begin:args.end]-stds[i][args.begin:args.end],color=colors[i],alpha=0.1)
plt.legend()
plt.savefig("rewards.png")