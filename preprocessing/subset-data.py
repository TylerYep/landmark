# Basically this snippet creates a subset of the training images from most occurring images for playground purposes

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# Subset selector for train dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import os
import const

# Counting occurences
df = pd.read_csv('data/train.csv')
df = df.replace(to_replace='None', value=np.nan).dropna()
df['landmark_id'] = df['landmark_id'].astype(int)

c = Counter(df['landmark_id'])
index, count = [], []

for i in c.items():
    index += [i[0]]
    count += [i[1]]

df_counts = pd.DataFrame(index=index, data=count, columns=['counts'])
df_counts = df_counts.sort_values('counts', ascending=False)
df_counts.to_csv('data/train-counts.csv')

def fetch_data(df):
    df_subset = []
    dict_counter = {}
    for row in tqdm(df.iterrows()):
        _id, url, landmark_id = row[1]
        if landmark_id in selected_index:
            if landmark_id not in dict_counter:
                dict_counter[landmark_id] = 1
                df_subset += [row[1]]
            elif dict_counter[landmark_id] < const.TAKE_N_OF_EACH:
                dict_counter[landmark_id] += 1
                df_subset += [row[1]]
        if all(value == const.TAKE_N_OF_EACH for value in dict_counter.values()):
            break
    return pd.DataFrame(df_subset)

# Find most occurring 500 unique images and take 10 of them
selected_index = df_counts.iloc[:const.N_MOST_FREQUENT_ELEMS, :].index

df_train = fetch_data(df)
df_train.to_csv('data/train-subset.csv', index=False)

# df = pd.read_csv('data/test.csv')
# df_test = fetch_data(df)
# df_test.to_csv('data/test-subset.csv', index=False)
