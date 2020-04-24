import os
import numpy as np
import json
from PIL import Image
import visualize
from matplotlib import pyplot as plt
import time



def normalization(matrix):
    '''
    Normalize the sum of elements in one 2D matrix to 1
    Input: 2D matrix
    Return: normalized 2D matrix
    '''

    norm = np.linalg.norm(matrix)
    matrix_new = np.copy(matrix)
    if norm >0:
        matrix_new = matrix_new / norm
    return matrix_new



def compute_convolution(I, T, stride=3, padding = False):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality.

    Return:
    heatmap: 3D array

    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    win_rows, win_cols = T.shape[0], T.shape[1]
    row_pad, col_pad = 0, 0
    if padding:
        if ((n_rows-win_rows) % stride != 0):
            row_pad = stride - (n_rows-win_rows) % stride

        if ((n_cols-win_cols) % stride !=0):
            col_pad = stride - (n_cols-win_cols) % stride

        I_new = np.zeros((n_rows+row_pad,n_cols+col_pad,n_channels))
        I_new[0:n_rows,0:n_cols,:] = np.copy(I)
        I = np.copy(I_new)
        (n_rows, n_cols, n_channels) = np.shape(I)

    heatmap = np.zeros((int((n_rows-win_rows)/stride+1),int((n_cols-win_cols)/stride+1),n_channels))
    for rows in range(0,n_rows-win_rows+1,stride):
        for cols in range(0,n_cols-win_cols+1,stride):
            for channel in range(n_channels):
                heatmap[int(rows/stride),int(cols/stride),channel] += np.sum(normalization(I[rows:rows+win_rows,cols:cols+win_cols,channel])*normalization(T[:,:,channel]))

    '''
    END YOUR CODE
    '''

    return heatmap



def recover_box(heatmap, coord, I, T, stride):
    '''
    With the heatmap in hand, we want to recover which box one particular pixel in heatmap comes from
    Arguments:
    heatmap: 3D array, height x width x channel
    coord: the coordinate pixel we care about
    I: the original image
    T: the template size

    return:
    a list of 4 ints; the bounding box
    '''

    row, col = coord[0], coord[1]
    l = col*stride
    r = min((col*stride+T.shape[1]),I.shape[1])
    t = row*stride
    b = min((row*stride+T.shape[0]),I.shape[0])
    return [t,l,b,r]






def predict_boxes(heatmap, I, T, stride=3, threshold = 0.9):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    output = []
    (h_row, h_col) = heatmap.shape
    for rows in range(h_row):
        for cols in range(h_col):
            if heatmap[rows][cols] > threshold:
                output.append((recover_box(heatmap,[rows,cols],I,T,stride) + [heatmap[rows][cols]]))

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I, T, strides=[3], padding=True, thres = 0.9, channel_sel = 'red'):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''

    channel_choice = {'red': 0, 'green': 1, 'blue': 2}

    if channel_sel == 'all':
        heatmap = compute_convolution(I, T, strides[0], padding)
        heatmap = (heatmap[:,:,0] + heatmap[:,:,1] + heatmap[:,:,2])/3

    else:
        I_new = I[:,:,channel_choice[channel_sel]].reshape((I.shape[0],I.shape[1],1))
        T_new = T[:,:,channel_choice[channel_sel]].reshape((T.shape[0],T.shape[1],1))
        heatmap = compute_convolution(I_new, T_new, strides[0], padding)[:,:,0]

    print(np.max(heatmap))

    output = predict_boxes(heatmap, I, T, stride=strides[0],threshold=thres)                             # The first convolution

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = 'D:\caltech new (Junior 2nd)\Courses\CS148\HW1\data\RedLights2011_Medium'

# load splits: 
split_path = 'D:\caltech new (Junior 2nd)\Courses\CS148\HW2\data\hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = 'D:\caltech new (Junior 2nd)\Courses\CS148\HW2\data\hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''


### Candidate Templates
#template_I = np.asarray(Image.open("D:\caltech new (Junior 2nd)\Courses\CS148\HW1\data\RedLights2011_Medium\RL-011.jpg"))
#T = template_I[68:100,350:378,:]
#template_I = np.asarray(Image.open("D:\caltech new (Junior 2nd)\Courses\CS148\HW1\data\RedLights2011_Medium\RL-096.jpg"))
#T = template_I[109:125, 287:297,:]
template_I = np.asarray(Image.open("D:\caltech new (Junior 2nd)\Courses\CS148\HW1\data\RedLights2011_Medium\RL-118.jpg"))
T = template_I[151:162,338:346,:]


### Training set outcome
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    ts = time.time()
    preds_train[file_names_train[i]] = detect_red_light_mf(I,T,strides=[3],padding=True,thres=0.95)

    ## Visualize result
    for box in preds_train[file_names_train[i]]:
        I = visualize.visualize_box(box[:4],I)
    visualize.visualize(I, address = "D:\caltech new (Junior 2nd)\Courses\CS148\HW2\data\images/result/"+file_names_train[i])

    te = time.time()
    print('Finish Image NO.%i; time elapsed: %.2f s' % (i, te - ts))




# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)


### Testing set outcome
if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        ts = time.time()
        preds_test[file_names_test[i]] = detect_red_light_mf(I, T, strides=[3], padding=True, thres=0.92)

        ## Visualize result
        for box in preds_test[file_names_test[i]]:
            I = visualize.visualize_box(box[:4], I)
        visualize.visualize(I, address="D:\caltech new (Junior 2nd)\Courses\CS148\HW2\data\images/result/" +
                                       file_names_test[i])
        te = time.time()
        print('Finish Image NO.%i; time elapsed: %.2f s' % (i, te - ts))

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
