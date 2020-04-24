import numpy as np
import os
import json

np.random.seed(2020) # to ensure you always get the same train/test split

data_path = 'D:\caltech new (Junior 2nd)\Courses\CS148\HW1\data\RedLights2011_Medium'
gts_path = 'D:\caltech new (Junior 2nd)\Courses\CS148\HW2\data\hw02_annotations'
split_path = 'D:\caltech new (Junior 2nd)\Courses\CS148\HW2\data\hw02_splits'
preds_path = 'D:\caltech new (Junior 2nd)\Courses\CS148\HW2\data\hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed


split_test = True # set to True and run when annotations are available

train_frac = 0.85

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# split file names into train and test
file_names_train = []
file_names_test = []
'''
Your code below. 
'''


order = np.arange(len(file_names)).astype(int)
np.random.shuffle(order)               # Create a random order
splitting = int(train_frac*len(file_names))
for i in range(splitting):
    file_names_train.append(file_names[order[i]])
for i in range(splitting,len(file_names),1):
    file_names_test.append(file_names[order[i]])

assert (len(file_names_train) + len(file_names_test)) == len(file_names)
assert len(np.intersect1d(file_names_train,file_names_test)) == 0

np.save(os.path.join(split_path,'file_names_train.npy'),file_names_train)
np.save(os.path.join(split_path,'file_names_test.npy'),file_names_test)

if split_test:
    with open(os.path.join(gts_path, "formatted_annotations_mturk.json"),'r') as f:
        gts = json.load(f)
    
    # Use file_names_train and file_names_test to apply the split to the
    # annotations
    gts_train = {}
    gts_test = {}
    '''
    Your code below. 
    '''
    train = np.load(os.path.join(split_path,'file_names_train.npy'))
    test = np.load(os.path.join(split_path,'file_names_test.npy'))
    for item in train:
        gts_train[item] = gts[item]
    for item in test:
        gts_test[item] = gts[item]


    with open(os.path.join(gts_path, 'annotations_train.json'),'w') as f:
        json.dump(gts_train,f)
    
    with open(os.path.join(gts_path, 'annotations_test.json'),'w') as f:
        json.dump(gts_test,f)
    
    
