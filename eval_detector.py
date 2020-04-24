import os
import json
import numpy as np
import visualize
import sys
from matplotlib import pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes- a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    i_l = max(box_1[0],box_2[0])  #union left bound
    i_r = min(box_1[2],box_2[2])  #union right bound
    i_t = max(box_1[1],box_2[1])  #union top bound
    i_b = min(box_1[3],box_2[3])  #union bottom bound
    w = max(i_r-i_l+1,0)
    h = max(i_b-i_t+1,0)
    i = h * w  # Intersection
    box1_area = (box_1[2]-box_1[0]+1)*(box_1[3]-box_1[1]+1)  # Box 1 area
    box2_area = (box_2[2]-box_2[0]+1)*(box_2[3]-box_2[1]+1) # Box 2 area
    u = 2*np.min([box1_area,box2_area]) - i # Union, which is consider to be 2x the smaller area
    iou = i/u
    
    assert (iou >= 0) and (iou <= 1.0)
    return iou



def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0


    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        pred_used = set()   #Make sure one pred to one ground truth
        for i in range(len(gt)):
            iou_max = -1
            iou_idx = -1
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                # Assign ground truth to the nearest pred
                # three conditions need to be satisfied:
                # (1) iou is the current maximum
                # (2) the pred is not assigned to other ground truth
                # (3) confidence exceeds the threshold
                if (iou > iou_max) and (j not in pred_used) and (pred[j][4] >= conf_thr):
                    iou_max = iou
                    iou_idx = j
            if iou_max >= iou_thr:
                TP += 1
                pred_used.add(iou_idx)
            else:
                FN += 1

        filtrate = 0
        for j in range(len(pred)):
            if pred[j][4] >= conf_thr:
                filtrate += 1
        FP += filtrate - len(pred_used)
        # Note that this algorithm may be problematic especially when two gt are close to each other.
        # The order of assigned used predictions may affect
        # Actual TP >= TP calculated.


    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = "D:\caltech new (Junior 2nd)\Courses\CS148\HW2\data\hw02_preds"
gts_path = "D:\caltech new (Junior 2nd)\Courses\CS148\HW2\data\hw02_annotations"

# load splits:
split_path = 'D:\caltech new (Junior 2nd)\Courses\CS148\HW2\data\hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train_rl096_small_red.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train_mturk.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test_rl096_small_red.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test_mturk.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 


# Use ascending confidence score as thresholds on training set
confidence_thrs = []
for fname in preds_train:
    if len(preds_train[fname]) > 0:
        for box in preds_train[fname]:
            confidence_thrs.append(box[4])
confidence_thrs = np.sort(np.array(confidence_thrs, dtype = float))

### Array used to calculate PR curve at iou = 0.25, 0.5, 0.75
tp_train25 = np.zeros(len(confidence_thrs))
fp_train25 = np.zeros(len(confidence_thrs))
fn_train25 = np.zeros(len(confidence_thrs))
tp_train50 = np.zeros(len(confidence_thrs))
fp_train50 = np.zeros(len(confidence_thrs))
fn_train50 = np.zeros(len(confidence_thrs))
tp_train75 = np.zeros(len(confidence_thrs))
fp_train75 = np.zeros(len(confidence_thrs))
fn_train75 = np.zeros(len(confidence_thrs))



for i, conf_thr in enumerate(confidence_thrs):
    tp_train25[i], fp_train25[i], fn_train25[i] = compute_counts(preds_train, gts_train, iou_thr=0.25, conf_thr=conf_thr)
    tp_train50[i], fp_train50[i], fn_train50[i] = compute_counts(preds_train, gts_train, iou_thr=0.5,
                                                                 conf_thr=conf_thr)
    tp_train75[i], fp_train75[i], fn_train75[i] = compute_counts(preds_train, gts_train, iou_thr=0.75,
                                                                 conf_thr=conf_thr)


# Plot training set PR curves
precision25 = tp_train25/(tp_train25+fp_train25)
recall25 = tp_train25/(tp_train25+fn_train25)
plt.plot(recall25,precision25,label = "iou = 0.25")
precision50 = tp_train50/(tp_train50+fp_train50)
recall50 = tp_train50/(tp_train50+fn_train50)
plt.plot(recall50,precision50,label = "iou = 0.5")
precision75 = tp_train75/(tp_train75+fp_train75)
recall75 = tp_train75/(tp_train75+fn_train75)
plt.plot(recall75,precision75,label = "iou = 0.75")
plt.xlabel('Recall',fontsize = 16)
plt.ylabel('Precision', fontsize = 16)
plt.title('PR Curve', fontsize = 18)
plt.legend(loc='best')
plt.show()




if done_tweaking:


    ### Array used to calculate PR curve at iou = 0.25, 0.5, 0.75
    tp_test25 = np.zeros(len(confidence_thrs))
    fp_test25 = np.zeros(len(confidence_thrs))
    fn_test25 = np.zeros(len(confidence_thrs))
    tp_test50 = np.zeros(len(confidence_thrs))
    fp_test50 = np.zeros(len(confidence_thrs))
    fn_test50 = np.zeros(len(confidence_thrs))
    tp_test75 = np.zeros(len(confidence_thrs))
    fp_test75 = np.zeros(len(confidence_thrs))
    fn_test75 = np.zeros(len(confidence_thrs))


    for i, conf_thr in enumerate(confidence_thrs):
        tp_test25[i], fp_test25[i], fn_test25[i] = compute_counts(preds_test, gts_test, iou_thr=0.25,
                                                                     conf_thr=conf_thr)
        tp_test50[i], fp_test50[i], fn_test50[i] = compute_counts(preds_test, gts_test, iou_thr=0.5,
                                                                     conf_thr=conf_thr)
        tp_test75[i], fp_test75[i], fn_test75[i] = compute_counts(preds_test, gts_test, iou_thr=0.75,
                                                                     conf_thr=conf_thr)

    # Plot training set PR curves
    precision25 = tp_test25[0:len(fp_test75)-1] / (tp_test25[0:len(fp_test75)-1] + fp_test25[0:len(fp_test75)-1])
    recall25 = tp_test25[0:len(fp_test75)-1] / (tp_test25[0:len(fp_test75)-1] + fn_test25[0:len(fp_test75)-1])
    plt.plot(recall25, precision25, label="iou = 0.25")
    precision50 = tp_test50[0:len(fp_test75)-1] / (tp_test50[0:len(fp_test75)-1] + fp_test50[0:len(fp_test75)-1])
    recall50 = tp_test50[0:len(fp_test75)-1] / (tp_test50[0:len(fp_test75)-1] + fn_test50[0:len(fp_test75)-1])
    plt.plot(recall50, precision50, label="iou = 0.5")
    precision75 = tp_test75[0:len(fp_test75)-1] / (tp_test75[0:len(fp_test75)-1] + fp_test75[0:len(fp_test75)-1])
    recall75 = tp_test75[0:len(fp_test75)-1] / (tp_test75[0:len(fp_test75)-1] + fn_test75[0:len(fp_test75)-1])
    plt.plot(recall75, precision75, label="iou = 0.75")
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.title('PR Curve of Test Set', fontsize=18)
    plt.legend(loc='best')
    plt.show()
