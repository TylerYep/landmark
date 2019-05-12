import numpy as np
import pandas as pd
import os

np.random.seed(2018)

base_dir = '../data/'
train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'))
submit_df = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))

# take the most frequent label
freq_label = train_df['landmark_id'].value_counts() / train_df['landmark_id'].value_counts().sum()

submit_df['landmarks'] = '%d %2.2f' % (freq_label.index[0], freq_label.values[0])
submit_df.to_csv('submission.csv', index=False)

r_idx = lambda : np.random.choice(freq_label.index, p = freq_label.values)
r_score = lambda idx: '%d %2.4f' % (freq_label.index[idx], freq_label.values[idx])

submit_df['landmarks'] = submit_df.id.map(lambda _: r_score(r_idx()))
submit_df.to_csv('rand_submission.csv', index=False)