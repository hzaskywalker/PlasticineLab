import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str,default=None)
args = parser.parse_args()

assert(args.dir!=None)
colors = ['red','blue','green','grey','orange','pink']

filenames = os.listdir('../'+args.dir)
if len(filenames)>6:
    print("At Most Can visualize 6 different Frame")
    exit(1)

dataframes = []
for fname in filenames:
    np_file = np.load('../'+args.dir+'/'+fname)
    x = np.arange(np_file.shape[1])
    df = pd.DataFrame(np_file.T,index=x.tolist(),columns=[fname for _ in range(np_file.shape[0])])
    dataframes.append(df)

for i in range(len(dataframes)):
    sns_plot = sns.lineplot(data=dataframes[i],palette=[colors[i]])
sns_plot.figure.savefig("rewards.png")


