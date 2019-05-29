# Basically this snippet creates a subset of the training images from most occurring images for playground purposes

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# Subset selector for train dataset
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import const
import logging

def get_counts(df):
    if os.path.exists('data/train-counts.csv'):
        logging.warning('Counts already exists. Skipping download.')
        return pd.read_csv('data/train-counts.csv')
    c = Counter(df['landmark_id'])
    index, count = [], []
    for i in c.items():
        index += [i[0]]
        count += [i[1]]

    df_counts = pd.DataFrame({'landmark_id': index, 'counts': count})
    df_counts = df_counts.sort_values('counts', ascending=False)
    df_counts.to_csv('data/train-counts.csv', index=None)
    return df_counts


def fetch_data(df, selected_index):
    df_subset = []
    dict_counter = dict.fromkeys(selected_index, 0) # iniitialize all landmark_id counters to 0
    for row in tqdm(df.iterrows()):
        _id, url, landmark_id = row[1]
        if landmark_id in selected_index:
            if dict_counter[landmark_id] < const.TAKE_N_OF_EACH:
                dict_counter[landmark_id] += 1
                df_subset += [row[1]]
        if all(value == const.TAKE_N_OF_EACH for value in dict_counter.values()):
            break
    return pd.DataFrame(df_subset)


if __name__ == '__main__':
    # Counting occurences
    df = pd.read_csv('data/train.csv')
    df = df.replace(to_replace='None', value=np.nan).dropna()
    df['landmark_id'] = df['landmark_id'].astype(int)

    # Find most occurring N_MOST_FREQUENT_ELEMS unique images and take TAKE_N_OF_EACH of them (see const.py)
    # df_counts = get_counts(df)
    # selected_index = list(df_counts[:const.N_MOST_FREQUENT_ELEMS]['landmark_id'])
    # df_train = fetch_data(df, selected_index)

    df_train = df.sample(const.TRAIN_SIZE)
    df_train.to_csv(const.TRAIN_CSV, index=False)

    df_dev = df.sample(const.DEV_SIZE)
    df_dev.to_csv(const.DEV_CSV, index=False)

    print(df_train.nunique())
    print(df_dev.nunique())
    print(pd.concat([df_train, df_dev]).nunique())
