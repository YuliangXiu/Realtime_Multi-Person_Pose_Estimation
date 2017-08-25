
# coding: utf-8

# In[1]:


import cv2 as cv 
import numpy as np
from numpy import loadtxt
from scipy.io import loadmat, savemat
import PIL.Image
from tqdm import tqdm, tqdm_notebook
import math
import caffe
import time
from config_reader import config_reader
import util
import copy
import os
from itertools import compress
import matplotlib
import pylab as plt


# In[2]:


# drawing parameters
mat_path = "/home/yuliang/data/MultiPerson_PoseTrack_v0.1/MultiPerson_PoseTrack_v0.1.mat"
bonn_mat = loadmat(mat_path)['RELEASE']
is_train = bonn_mat['is_train'][0,0]

video_names = [video_name[0][0] for video_name in bonn_mat['annolist'][0,0]['name']]
video_frames = [video_frame[0][0] for video_frame in bonn_mat['annolist'][0,0]['num_frames']]

train_names = list(compress(video_names[:], is_train))
test_names = [x for x in video_names if x not in train_names]

train_frames = [item[0] for item in list(compress(video_frames[:], is_train))]
test_frames = [item[0] for item in list(compress(video_frames[:], 1-is_train))]


# In[4]:


param, model = config_reader()
root_dir = "/home/yuliang/data/MultiPerson_PoseTrack_v0.1/videos/"
image_x, image_y, image_channel = cv.imread(os.path.join(root_dir,'000002','00001.jpg')).shape

oriImgs = dict()
multiplier = dict()

for vid, test_name in enumerate(tqdm_notebook(test_names)):
    oriImgs[test_name] = dict()
    for frame_name in ['{:0>5}.jpg'.format(frame+1) for frame in range(test_frames[vid])]:
        oriImgs[test_name][frame_name] = cv.imread(os.path.join(root_dir,test_name,frame_name))
        image_x, image_y, channel = oriImgs[test_name][frame_name].shape
        multiplier[test_name] = [x * model['boxsize'] /image_x for x in param['scale_search']]


# In[5]:


if param['use_gpu']: 
    caffe.set_mode_gpu()
    caffe.set_device(param['GPUdeviceNumber']) # set to your device!
else:
    caffe.set_mode_cpu()
net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)


# In[ ]:


heatmap = dict()
paf = dict()

part = 18  # all parts

for vid, test_name in enumerate(tqdm_notebook(test_names)):
    heatmap[test_name] = dict()
    paf[test_name] = dict()
    for fid, frame_name in enumerate(['{:0>5}.jpg'.format(frame+1) for frame in range(test_frames[vid])]):
        image_x, image_y, channel = oriImgs[test_name][frame_name].shape
        heatmap[test_name][frame_name] = dict()
        paf[test_name][frame_name] = dict()
        
        heatmap[test_name][frame_name]['heatmap_avg'] = np.zeros((image_x, image_y, 19))
        paf[test_name][frame_name]['paf_avg'] = np.zeros((image_x, image_y, 38))
        for m in range(len(multiplier[test_name])):
            scale = multiplier[test_name][m]
            heatmap[test_name][frame_name][scale] = dict()
            paf[test_name][frame_name][scale] = dict()
            
            imageToTest = cv.resize(oriImgs[test_name][frame_name], (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model['stride'], model['padValue'])

            net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
            net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
            output_blobs = net.forward()

            # extract outputs, resize, and remove padding
            heatmap_ = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1,2,0)) # output 1 is heatmaps
            heatmap_ = cv.resize(heatmap_, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
            heatmap_ = heatmap_[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
            heatmap[test_name][frame_name][scale] = cv.resize(heatmap_, (image_y, image_x), interpolation=cv.INTER_CUBIC)
            heatmap[test_name][frame_name]['heatmap_avg'] = heatmap[test_name][frame_name]['heatmap_avg'] + heatmap[test_name][frame_name][scale] / len(multiplier[test_name])

            # extract pad        
            paf_ = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1,2,0)) # output 0 is PAFs
            paf_ = cv.resize(paf_, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
            paf_ = paf_[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
            paf[test_name][frame_name][scale] = cv.resize(paf_, (image_y, image_x), interpolation=cv.INTER_CUBIC)
            paf[test_name][frame_name]['paf_avg'] = paf[test_name][frame_name]['paf_avg'] + paf[test_name][frame_name][scale] / len(multiplier[test_name])

    np.save(test_name+'_heatmap.npy', heatmap[test_name])
    np.save(test_name+'_paf.npy', paf[test_name])
    heatmap[test_name] = dict()
    heatmap[test_name]['npy_path'] = test_name+'_heatmap.npy'
    paf[test_name] = dict()
    paf[test_name]['npy_path'] = test_name+'_paf.npy'

# In[6]:


import scipy
from scipy.ndimage.filters import gaussian_filter

all_peaks = dict()
peak_counter = dict()

for vid, test_name in enumerate(tqdm_notebook(test_names)):
    all_peaks[test_name] = dict()
    peak_counter[test_name] = dict()
    for fid, frame_name in enumerate(['{:0>5}.jpg'.format(frame+1) for frame in range(test_frames[vid])]):
        all_peaks[test_name][frame_name] = []
        peak_counter[test_name][frame_name] = 0
        for part in range(18):
            x_list = []
            y_list = []
            map_ori = heatmap[test_name][frame_name]['heatmap_avg'][:,:,part]
            map = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map.shape)
            map_left[1:,:] = map[:-1,:]
            map_right = np.zeros(map.shape)
            map_right[:-1,:] = map[1:,:]
            map_up = np.zeros(map.shape)
            map_up[:,1:] = map[:,:-1]
            map_down = np.zeros(map.shape)
            map_down[:,:-1] = map[:,1:]

            peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
            peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
            id = range(peak_counter[test_name][frame_name], peak_counter[test_name][frame_name] + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks[test_name][frame_name].append(peaks_with_score_and_id)
            peak_counter[test_name][frame_name] += len(peaks)


# In[7]:


# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10],[10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], [1,16], [16,18], [3,17], [6,18]]
# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22],[23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52],[55,56], [37,38], [45,46]]


# In[8]:


connection_all = dict()
special_k = dict()
mid_num = 10

for vid, test_name in enumerate(tqdm_notebook(test_names)):
    for fid, frame_name in enumerate(['{:0>5}.jpg'.format(frame+1) for frame in range(test_frames[vid])]):
        connection_all[test_name][frame_name] = []
        special_k[test_name][frame_name] = []
        image_x, image_y, channel = oriImgs[test_name][frame_name].shape

        for k in range(len(mapIdx)):
            score_mid = paf[test_name][frame_name]['paf_avg'][:,:,[x-19 for x in mapIdx[k]]]
            candA = all_peaks[test_name][frame_name][limbSeq[k][0]-1]
            candB = all_peaks[test_name][frame_name][limbSeq[k][1]-1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if(nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                        vec = np.divide(vec, norm)

                        startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),np.linspace(candA[i][1], candB[j][1], num=mid_num))

                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*image_x/norm-1, 0)
                        criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0,5))
                for c in range(len(connection_candidate)):
                    i,j,s = connection_candidate[c][0:3]
                    if(i not in connection[:,3] and j not in connection[:,4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if(len(connection) >= min(nA, nB)):
                            break

                connection_all[test_name][frame_name].append(connection)
            else:
                special_k[test_name][frame_name].append(k)
                connection_all[test_name][frame_name].append([])


# In[9]:


# last number in each row is the total parts number of that person
# the second last number in each row is the score of the overall configuration
subset = dict()
candidate = dict()
for vid, test_name in enumerate(tqdm_notebook(test_names)):
    candidate[test_name] = dict()
    subset[test_name] = dict()
    for fid, frame_name in enumerate(['{:0>5}.jpg'.format(frame+1) for frame in range(test_frames[vid])]):
        image_x, image_y, channel = oriImgs[test_name][frame_name].shape
        candidate[test_name][frame_name] = np.array([item for sublist in all_peaks[test_name][frame_name] for item in sublist])
        subset[test_name][frame_name] = -1 * np.ones(( 0, 20))
        for k in range(len(mapIdx)):
            if k not in special_k[test_name][frame_name]:
                partAs = connection_all[test_name][frame_name][k][:,0]
                partBs = connection_all[test_name][frame_name][k][:,1]
                indexA, indexB = np.array(limbSeq[k]) - 1
                for i in range(len(connection_all[test_name][frame_name][k])): #= 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset[test_name][frame_name])): #1:size(subset,1):
                        if subset[test_name][frame_name][j][indexA] == partAs[i] or subset[test_name][frame_name][j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if(subset[test_name][frame_name][j][indexB] != partBs[i]):
                            subset[test_name][frame_name][j][indexB] = partBs[i]
                            subset[test_name][frame_name][j][-1] += 1
                            subset[test_name][frame_name][j][-2] += candidate[test_name][frame_name][partBs[i].astype(int), 2] + connection_all[test_name][frame_name][k][i][2]
                    elif found == 2: # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        print "found = 2"
                        membership = ((subset[test_name][frame_name][j1]>=0).astype(int) + (subset[test_name][frame_name][j2]>=0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0: #merge
                            subset[test_name][frame_name][j1][:-2] += (subset[test_name][frame_name][j2][:-2] + 1)
                            subset[test_name][frame_name][j1][-2:] += subset[test_name][frame_name][j2][-2:]
                            subset[test_name][frame_name][j1][-2] += connection_all[test_name][frame_name][k][i][2]
                            subset[test_name][frame_name] = np.delete(subset[test_name][frame_name], j2, 0)
                        else: # as like found == 1
                            subset[test_name][frame_name][j1][indexB] = partBs[i]
                            subset[test_name][frame_name][j1][-1] += 1
                            subset[test_name][frame_name][j1][-2] += candidate[test_name][frame_name][partBs[i].astype(int), 2] + connection_all[test_name][frame_name][k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[test_name][frame_name][connection_all[test_name][frame_name][k][i,:2].astype(int), 2]) + connection_all[test_name][frame_name][k][i][2]
                        subset[test_name][frame_name] = np.vstack([subset[test_name][frame_name], row])


# In[10]:


# delete some rows of subset which has few parts occur
for vid, test_name in enumerate(tqdm_notebook(test_names)):
    for fid, frame_name in enumerate(['{:0>5}.jpg'.format(frame+1) for frame in range(test_frames[vid])]):
        deleteIdx = [];
        for i in range(len(subset[test_name][frame_name])):
            if subset[test_name][frame_name][i][-1] < 4 or subset[test_name][frame_name][i][-2]/subset[test_name][frame_name][i][-1] < 0.4:
                deleteIdx.append(i)
        subset[test_name][frame_name] = np.delete(subset[test_name][frame_name], deleteIdx, axis=0)


# In[15]:


# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], [1,16], [16,18], [3,17], [6,18]]
# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], [55,56], [37,38], [45,46]]

np.save('subset.npy', subset)
np.save('all_peaks.npy', all_peaks)
np.save('candidate.npy', candidate)
np.save('oriImgs.npy',oriImgs) 
np.save('multiplier.npy',multiplier)
np.save('heatmap.npy',heatmap)
np.save('paf.npy',paf)
np.save('peak_counter.npy',peak_counter)
np.save('connection_all.npy',connection_all)
np.save('special_k.npy',special_k)


# In[ ]:


# # optical flow 
# all_flow = np.zeros((frame_num, oriImg.shape[1], oriImg.shape[2], 2))

# # show optical flow
# f3, axarr3 = plt.subplots(rows_small, cols_small)
# f3.set_size_inches((width_small, height_small))

# frame1 = oriImg[0]
# prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame1)
# hsv[...,1] = 255

# for frame in tqdm_notebook(range(1,frame_num)):
#     frame2 = oriImg[frame]
#     next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)

#     all_flow[frame] = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#     mag, ang = cv.cartToPolar(all_flow[frame][...,0], all_flow[frame][...,1])
#     hsv[...,0] = ang*180/np.pi/2
#     hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
#     rgb = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

#     axarr3[frame/cols_small][frame%cols_small].imshow(rgb)
#     prvs = next

