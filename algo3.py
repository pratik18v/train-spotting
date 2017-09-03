#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 05:13:48 2017

@author: pratik18v
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:12:58 2017

@author: pratik18v

"""

import numpy as np
import cv2
import glob
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby
from scipy.signal import medfilt
from peakdetect import peakdetect

#-----------------------------------------------------------------------------
#Finding region area vertically
folder = 'train_crossing_3/opt_flow_3/full_res/'
im = cv2.imread(folder + 'opticalhsv0.png', 0)
[m,n] = im.shape
avg_im = np.zeros([m,n])
ctr = 0
for fname in glob.glob(folder+'*.png'):
    avg_im += cv2.imread(fname, 0)
    ctr += 1
avg_im = avg_im/ctr

ret2,th2 = cv2.threshold(np.asarray(avg_im, dtype=np.uint8),0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(th2,kernel,iterations = 1)
dilation = cv2.dilate(th2,kernel,iterations = 1)
kernel = np.ones((50,50),np.uint8)
closing = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
row_sums = np.sum(closing, axis=1).tolist()
for rs in row_sums:
    if int(rs) > 0.5*480:
        horz_bndry_1 = row_sums.index(rs)
        break
        
for rs in row_sums[::-1]:
    if int(rs) > 0.5*1280:
        horz_bndry_2 = row_sums.index(rs)
        break
#-----------------------------------------------------------------------------
num_files = 0
#folder = 'train_crossing_3/back_sub_3/'
folder = 'train_crossing_3/blob_output/full_res/'
N = 5
w = 50
hist_length = 800
rows = [np.zeros([hist_length])]
hists = []
flag = False
aligned = False
corr_threshold = 10
def save_plot(save_dir, ctr, row_thresh):
    if row_thresh.min() != row_thresh.max():
        save_path = save_dir + 'frame' + str(ctr) + '.jpg'
        plt.plot(row_thresh)
        plt.savefig(save_path)
        plt.close()
        #print 'Saved: {}'.format(save_path) 

def sim_measure(vec1, vec2):
    return pow(sum(pow((vec1 - vec2), 2)), 0.5)/ len(vec1)

def grouper(data):
    groups = []
    for k, g in groupby(enumerate(data), lambda (i,x):i-x):
        groups.append(map(itemgetter(1), g))

    return groups
fnums = []
for fname in glob.glob(folder+'*.jpg'):
    num_files += 1
    fnums.append(int(fname.split('.')[0].split('frame')[-1]))

num_frames = num_files
pred_shift = 0
rel_shifts = []
pred_shifts = []
final_shifts = []
pred_speed = []
final_corrs = []
fnums.sort()
ignore_nums = 0
fcnt = 0
invalids = []
temp1 = []
temp2 = []
for i in fnums:
    print 'Frames processed: {}'.format(i)

    aligned = False
    #Get blob image
    bkg = cv2.imread(folder + 'frame' + str(i) + '.jpg', 0)

    corrs = []
    shifts = []

    #Generate histogram of white pixels (first hist_length pixels)
    row_thresh = np.sum(bkg, axis=0) / 255
    #row_thresh[800:875] = row_thresh[799]
    row_thresh = row_thresh[:hist_length]
    row_thresh = row_thresh[::-1]
    #TEMP BUG FIX: last element always zero
    row_thresh[-1] = row_thresh[-2]
    
    row_thresh[row_thresh[:] > 20] = 20

    #save_plot('temp/', i, row_thresh)                       
    #Ignore flat histograms
    if row_thresh.min() == row_thresh.max():
        ignore_nums += 1
        invalids.append(i)
        # continue

#    valid_fnums = [x for x in fnums if x not in invalids]
    if len(hists) >= 1:
        prev_hist = hists[-1]
        curr_hist = row_thresh

        if len(final_shifts) == 1:
            rel_shifts.append(final_shifts[-1])
        elif len(final_shifts) > 1 and len(final_shifts) <= N:
            rel_shifts.append(final_shifts[-1] - final_shifts[-2])
        elif len(final_shifts) > N:
            #rel_shift = sum(rel_shifts[-N:])/N
            rel_shift = int(np.mean(rel_shifts)+0.5)
            pred_speed.append(rel_shift)
            #rel_shift = int(np.ma.average(rel_shifts[-N:]))
            pred_shift = (len(prev_hist) - hist_length) + rel_shift
            #rel_shifts.append(rel_shift)
            pred_shifts.append(pred_shift)

        for j in range(-w, w+1):
            shift = pred_shift + j
            #print len(curr_hist) + shift, len(prev_hist), signal_length
            if shift > 0:
                signal_length = min(len(curr_hist) + shift, len(prev_hist))
                padding = np.zeros([abs(shift)])
                curr_aligned_hist = np.hstack((padding, curr_hist))[:signal_length]
                prev_aligned_hist = prev_hist[:signal_length]
            else:
                curr_aligned_hist = curr_hist[abs(shift):]
                prev_aligned_hist = prev_hist
                padding = np.zeros([len(prev_aligned_hist) - len(curr_aligned_hist)])
                curr_aligned_hist = np.hstack((curr_aligned_hist, padding))
            #print len(curr_hist), len(prev_hist), shift, len(padding), signal_length
            corr = sim_measure(prev_aligned_hist, curr_aligned_hist)
            corrs.append(corr)
            shifts.append(shift)
#            plt.plot(prev_aligned_hist)
#            plt.plot(curr_aligned_hist)
#            plt.legend(['prev', 'current'])
#            plt.title(str(shift) + ' ' + str(corr))
#            plt.savefig('train_crossing_3/alignments_3/two_hist_' + str(i) + '_' + str(shift+w) + '.jpg')
#            plt.close()
#        if min(corrs) <=  corr_threshold:
#            temp1.append(min(corrs))
#            temp2.append(shifts[corrs.index(min(corrs))])
#            final_shift = pred_shift
#        else:
#            final_shift = shifts[corrs.index(min(corrs))]
        final_shift = shifts[corrs.index(min(corrs))]
        final_corrs.append(min(corrs))
        if len(final_shifts) > 0:
            if final_shift - final_shifts[-1] <= corr_threshold:
                final_shift = pred_shift
            else:
                temp1.append(final_shift)
        final_shifts.append(final_shift)

        if final_shift > 0:
            padding = np.zeros([final_shift])
            max_aligned = np.hstack((padding, curr_hist))
        else:
            curr_aligned_hist = curr_hist[abs(final_shift):]
            prev_aligned_hist = prev_hist
            padding = np.zeros([len(prev_aligned_hist) - len(curr_aligned_hist)])
            max_aligned = np.hstack((curr_aligned_hist, padding))
        aligned = True

        if len(final_shifts) > N:
            rel_shifts.append(final_shifts[-1] - final_shifts[-2])
        aligned = True

#        if len(rel_shifts) >= 1:
#            plt.figure(figsize=(20, 4))
#            if len(max_aligned) <= 1000:
#                scale = 200
#            elif len(max_aligned) > 1000 and len(max_aligned) <= 5000:
#                scale = 500
#            elif len(max_aligned) > 5000 and len(max_aligned) <= 10000:
#                scale = 1000
#            plt.xticks(np.arange(0, len(max_aligned)+1, scale))
#            plt.plot(prev_hist)    
#            plt.plot(max_aligned)
#            plt.legend(['prev', 'current'])
#            plt.savefig('train_crossing_3/alignments_3/two_hist' + str(i) + '_' + str(rel_shifts[-1]) + '_' + str(min(corrs)) + '.jpg')
#            plt.close()

    rows.append(row_thresh)
    if aligned == True:
        hists.append(max_aligned)
    else:
        hists.append(row_thresh)

    fcnt += 1

valid_fnums = [x for x in fnums if x not in invalids]

#Taking union
start = [0]
end = [hist_length]
for i in range(len(final_shifts)):
    next_start = final_shifts[i]
    next_end = next_start + hist_length
    start.append(next_start) 
    end.append(next_end)
    
lens = [len(h) for h in hists]
union = np.zeros([max(lens)])
divisors = np.zeros([max(lens)])

for i in range(len(hists)):
    #hists[i][hists[i][:] > 20] = 50
    union[:len(hists[i])] += hists[i]

for i in range(max(lens)):
    for j in range(len(start)):
        if i >= start[j] and i <= end[j]:
            divisors[i] += 1

union_norm = np.asarray(union) / np.asarray(divisors)
#union_norm = union
save_folder = 'figs/full_res/'
#Plots
str_version = 'frames-{}_N-{}_w-{}_corr-{}'.format(num_frames, N, w, corr_threshold)
#1. Final histogram
plt.figure(figsize=(20, 4))
plt.plot(union_norm)
plt.xlabel('row pixel')
plt.ylabel('avg number of white pixels')
plt.savefig(save_folder + 'union_hist_' + str_version + '.jpg')
plt.close()

#2. Speed vs frame number
plt.figure(figsize=(20, 4))
plt.plot(fnums[1:], rel_shifts, 'bo', markersize=1)
plt.xlabel('frame number')
plt.ylabel('speed')
plt.savefig(save_folder + 'speed_vs_frame_no_' + str_version + '.jpg')
plt.close()

#3. Actual vs predicted shifts
plt.figure(figsize=(20, 4))
plt.plot(np.asarray(pred_shifts)/100)
plt.plot(np.asarray(final_shifts[-len(pred_shifts):])/100)
plt.legend(['predicted', 'actual'])
plt.ylabel('Overall shift')
plt.xlabel('frame number')
plt.savefig(save_folder + 'pred_vs_actual_shifts_' + str_version + '.jpg')
plt.close()

#4. Difference in actual and predicted shifts
plt.figure(figsize=(20, 4))
plt.plot(np.asarray(final_shifts[-len(pred_shifts):]) - np.asarray(pred_shifts), 'bo', markersize=1)
plt.ylabel('Offset')
plt.xlabel('frame number')
plt.savefig(save_folder + 'diff_shifts_' + str_version + '.jpg')
plt.close()

#Generating crops

#smooth_union = pd.Series(union_norm).rolling(50).mean()
#pred_gaps = argrelextrema(np.asarray(union_norm), np.less, order=70)
#pred_gaps = np.insert(pred_gaps, 0, np.where(union_norm != 0)[0][0])
#pred_gaps = np.insert(pred_gaps, len(pred_gaps), np.where(union_norm != 0)[0][-1])
smooth_window = 15
#df_hist = pd.DataFrame({'raw': union_norm})
#df_hist['smooth'] = df_hist.rolling(smooth_window).mean()
#smooth_union = df_hist['smooth'].as_matrix()[smooth_window:]
smooth_union = medfilt(union_norm, smooth_window)
#smooth_union = union_norm
_, _min = peakdetect(smooth_union,lookahead=10, delta=3)
pred_gaps = np.asarray([p[0] for p in _min])
yn = np.asarray([p[1] for p in _min])
pred_gaps = np.insert(pred_gaps, 0, np.where(smooth_union != 0)[0][0])
pred_gaps = np.insert(pred_gaps, len(pred_gaps), np.where(smooth_union != 0)[0][-1])
yn = np.insert(yn, 0, 0)
yn = np.insert(yn, len(yn), 0)

#5. Plotting minima points in union histogram
pt_no = np.linspace(0,len(pred_gaps)-1, len(pred_gaps)).astype(np.uint8)
fig = plt.figure(figsize=(30,6))
plt.xticks(np.arange(0, len(smooth_union)+1, 1000))
ax = fig.add_subplot(111)
ax.grid()
plt.plot(smooth_union)
plt.scatter(pred_gaps, yn, color = 'red')
for i, txt in enumerate(pt_no):
    ax.annotate(txt, (pred_gaps[i],max(0, yn[i])))
plt.savefig(save_folder + 'union_minimas_' + str_version + '.jpg')
plt.close()

#diffs = []
#inval_gaps = []
#for i in range(len(pred_gaps)-1):
#    diff = pred_gaps[i+1] - pred_gaps[i]
#    if diff < 150:
#        inval_gaps.append(i+1)
#    diffs.append(diff)
#
#for ig in inval_gaps:
#    pred_gaps = np.delete(pred_gaps, ig)
#    yn = np.delete(yn, ig)
#    pt_no = np.delete(pt_no, ig)

#req_frames = []
#req_start = []
#req_end = []
#car_lengths = []
#for i in range(1,len(pred_gaps)):
#    for j in range(len(end)):
#        if end[j] > pred_gaps[i]:
#            req_frames.append(fnums[j])
#            req_start.append(end[j] - pred_gaps[i])
#            car_lengths.append(pred_gaps[i] - pred_gaps[i-1])
#            req_end.append(req_start[-1] + car_lengths[-1])
#            break
#
#for i in range(len(req_frames)):
#    im = cv2.imread('train_crossing_3/frames/full_res/frame' + str(req_frames[i]) + '.jpg')
#    crop_im = im[horz_bndry_1:horz_bndry_2,max(0, req_start[i]-10):req_end[i]+10,:]
#    cv2.imwrite('train_crossing_3/cropped/full_res/car_' + str(i) + '.jpg', crop_im)

#df = pd.DataFrame({'frame_no': req_frames, 'car_length': diffs})
#df = df[['frame_no', 'car_length']]
#df.to_html(save_folder + 'car_info.html')


#x = 100
#avg_speed = []
#for i in range(len(pred_speed)):
##    pt_a = max(0, i-x)
##    pt_b = min(i+x, len(pred_speed)-1)
##    window = pred_speed[pt_a:i] + pred_speed[i+1:pt_b]
#    window = pred_speed[:i]
#    avg_speed.append(np.mean(window))
#
#plt.figure(figsize=(20,4))
#plt.plot(avg_speed)
#plt.savefig('figs/temp_pred.jpg')
#plt.close()


