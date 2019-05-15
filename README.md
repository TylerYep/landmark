# Landmark Recognition
#### CS 230 Project

Main Challenge:
https://www.kaggle.com/c/landmark-retrieval-2019/overview

Baseline Model:
https://www.kaggle.com/c/landmark-recognition-challenge/discussion/57919

### Step 1: Download Dataset CSV Link
https://www.kaggle.com/c/landmark-retrieval-2019/data
The above link contains csv files with links to all of the images for the train and test sets. Unzip the folder and put it into google-landmarks-dataset/, and then specify the number of examples you want to download in const.py. You can also manually change whether you want to download from the train or set set.

## Step 1.5: Conda Install
Run ``` conda env create -f ennviroment.yml ```.

### Step 2: Get Subset of Data
Run ``` python preprocessing/subset-data.py ```.

(Note: everything should be run from the ```landmark/``` level.)

This file outputs a modified ```train-subset.csv``` file to fetch images from. You can specify how many unique landmarks you want and how many of each you want by changing variables in ```const.py```.

### Step 3: Download Images
Run ``` python download-images.py ```.

Hopefully this doesn't take forever. If you simply want all of the images, use the .sh file or download from a link on the Kaggle page.


pip install cnn_finetune

## To ask TAs:
- Do we still want data augmentation when we have too much training data?


