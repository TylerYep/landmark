import pandas as pd
from pathlib import Path

df = pd.read_csv("data/train-old.csv")

print (df.head())
data_path = "data/images/train/"
img_exists = df['id'].map(lambda x: Path(data_path + x + ".jpg").is_file() * 1.0)


print ("total examples: " + str(len(df)))
print ("missing examples: " + str(len(df) - sum(img_exists)))

df = df[img_exists == 1]
print (len(df)) 

df.to_csv("data/train.csv")




