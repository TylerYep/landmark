# Landmark Recognition
#### CS 230 Project

Main Challenge:
https://www.kaggle.com/c/landmark-retrieval-2019/overview

Baseline Model:
https://www.kaggle.com/c/landmark-recognition-challenge/discussion/57919

## Step 1: Install Conda Ennviroment
Run ``` conda env create -f ennviroment.yml ```.

### Step 2: Download Dataset CSV Link
https://www.kaggle.com/c/landmark-retrieval-2019/data
The above link contains csv files with links to all of the images for the train and test sets. Unzip the folder and put it into data/images/, and then specify the number of examples you want to download in const.py. You can also manually change whether you want to download from the train, dev, or test set.

### Step 3: Get Subset of Data
Run ``` python preprocessing/subset-data.py ```.

(Note: everything should be run from the ```landmark/``` level.)

This file outputs a modified ```train-subset.csv``` file to fetch images from. You can specify how many unique landmarks you want and how many of each you want by changing variables in ```const.py```.

### Step 3: Download Images
Run ``` python download-images.py ```.

Hopefully this doesn't take forever. If you simply want all of the images, use the .sh file or download from a link on the Kaggle page.

## Workflow
Basically run train.py, which currently relies on three places: dataset, const, and layers.

dataset.py
const.py
train.py
test.py
util.py




## To ask TAs:
- Do we still want data augmentation when we have too much training data?


