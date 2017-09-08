#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 09:29:39 2017

@author: pratik18v
"""
import time
import tqdm
import numpy as np
import cv2
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby
from scipy.signal import medfilt
from peakdetect import peakdetect

#Auxillary functions ##########################################################
def sim_measure(vec1, vec2, i, j):
    corr = pow(sum(pow((vec1 - vec2), 2)), 0.5)/ len(vec1)
    plt.plot(vec1)
    plt.plot(vec2)
    plt.legend(['current', 'prev'])
    plt.savefig(stem + 'alignments/two_hist_' + str(i) + '_' + str(j) + '_' + str(corr) + '.jpg')
    plt.close()
    return corr

def grouper(data):
    groups = []
    for k, g in groupby(enumerate(data), lambda (i,x):i-x):
        groups.append(map(itemgetter(1), g))
    return groups
###############################################################################

#Initializations ##############################################################
#1. BB_Norfolk_Southern_WB.MOV : https://drive.google.com/open?id=0B8EK0LxdsI4qMUJNZHdheGl0LUU
#2. BB_NS_WB : https://drive.google.com/open?id=0B8EK0LxdsI4qdjJhR0NGaUt1eXc
#3. BB_NS_mixed_freight : https://drive.google.com/open?id=0B8EK0LxdsI4qYUpxYzBKY05kSVE
#4. train_crossing_3 : https://drive.google.com/open?id=0B3_7z0wYZNLQMnpvUHowNmFpcUU
#5. train_crossing_4 : https://drive.google.com/open?id=0B3_7z0wYZNLQS3d2TW5mTFQwU0E

video_name = 'BB_NS_WB.MOV'
video_link = 'https://drive.google.com/open?id=0B8EK0LxdsI4qdjJhR0NGaUt1eXc'
stem = video_name.split('.')[0] + '/'
folder1 = stem + 'opt_flow/'
folder2 = stem + 'new_backsub/'
graph_dir = 'graphs/'
hist_length = 1920
union_length = int(1.5e5)
bs_N = 10
bs_thresh = 30
N = 5
w = 100
intensity_thresh = 30
smooth_window = 5
lookahead = 5
delta = 5
###############################################################################

#Part 0: Background subtraction ###############################################
bkg_im = cv2.imread(stem + 'frames/frame0.jpg', 0).astype(float)
[m,n] = bkg_im.shape

for i in range(1,bs_N):
    bkg_im += cv2.imread(stem + 'frames/frame' + str(i) + '.jpg', 0).astype(float)
    
bkg_im /= bs_N
#plt.imshow(bkg_im.astype(np.uint8))

num_files = 0
for fname in glob.glob(stem + 'frames/*.jpg'):
    num_files += 1

for i in tqdm(range(bs_N, num_files)):
    diff = np.abs(bkg_im - \
                     cv2.imread(stem + 'frames/frame' + str(i) + '.jpg', 0))
    blur = cv2.GaussianBlur(diff.astype(np.uint8),(11,11),0)
    th1 = (blur > bs_thresh) * 255
    #th2 = np.max(th1, 2)
    #th = cv2.adaptiveThreshold(diff.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY,11,2)
    #ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite(folder2 + 'frame' + str(i) + '.jpg', th1)

###############################################################################

#Part 1: Find horizontal boundaries ###########################################
start_time = time.time()
im = cv2.imread(folder1 + 'opticalhsv0.png', 0)
[m,n] = im.shape
avg_im = np.zeros([m,n])
ctr = 0

print 'Computing train region across vertical-axis ... '
for fname in glob.glob(folder1 + '*.png'):
    avg_im += cv2.imread(fname, 0)
    ctr += 1
avg_im = avg_im/ctr

ret2,th2 = cv2.threshold(np.asarray(avg_im, dtype=np.uint8),0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(th2,kernel,iterations = 2)
dilation = cv2.dilate(erosion,kernel,iterations = 8)
row_sums = np.sum(dilation, axis=1).tolist()
for rs in row_sums:
    if int(rs) > 0.5*n:
        horz_bndry_1 = row_sums.index(rs)
        break
        
for rs in row_sums[::-1]:
    if int(rs) > 0.5*n:
        horz_bndry_2 = m - row_sums[::-1].index(rs)
        break
end_time = time.time()
print 'DONE'
dur1 = end_time - start_time
print 'Duration: {} sec'.format(dur1)
###############################################################################


#Part 2: Check direction ######################################################
start_time = time.time()
fnums = []
for fname in glob.glob(folder2 + '*.jpg'):
    fnums.append(int(fname.split('.')[0].split('frame')[-1]))
fnums.sort()

print 'Computing direction of train ... '
row_hist = np.zeros(hist_length)
i = fnums[0]
while len(np.where(row_hist != 0)[0]) <= n/4:
    i += 1
    bkg = cv2.imread(folder2 + 'frame' + str(i) + '.jpg', 0)
    row_hist = np.sum(bkg, axis=0) / 255
    
if max(np.where(row_hist != 0)[0]) > n/2:
    direc = 'R-L'
else:
    direc = 'L-R'

end_time = time.time()
print 'DONE'
dur2 = end_time - start_time
print 'Duration: {}'.format(dur2)
print 'Direction of train: {}'.format(direc)
###############################################################################

#Part 3: Alignment ############################################################
start_time = time.time()
final_shifts = []
pred_shifts = [0]

start = [0]
end = [hist_length]

#Intializing global histogram
hist_union = np.zeros(union_length)
bkg = cv2.imread(folder2 + 'frame' + str(fnums[0]) + '.jpg', 0)
row_hist = np.sum(bkg, axis=0) / 255
row_hist = row_hist[:hist_length]
row_hist[row_hist[:] > intensity_thresh] = intensity_thresh
row_hist[row_hist[:] < intensity_thresh] = 0
#row_hist = medfilt(row_hist, 5)
#If direction L-R, reverse histogram to keep head at origin (in global histogram)    
if direc.startswith('L'):
    row_hist = row_hist[::-1]
    #TEMP BUG FIX: last element always zero
    row_hist[-1] = row_hist[-2]

hist_union[start[-1]:end[-1]] += row_hist

print 'Starting alignment ... '

for i in tqdm.tqdm(range(1,len(fnums))):

    bkg = cv2.imread(folder2 + 'frame' + str(fnums[i]) + '.jpg', 0)
    curr_hist = np.sum(bkg, axis=0) / 255
    curr_hist = curr_hist[:hist_length]
    #If direction L-R, reverse histogram to keep head at origin (in global histogram)    
    if direc.startswith('L'):
        curr_hist = curr_hist[::-1]
        #TEMP BUG FIX: last element always zero
        curr_hist[-1] = curr_hist[-2]

    curr_hist[curr_hist[:] > intensity_thresh] = intensity_thresh
    curr_hist[curr_hist[:] < intensity_thresh] = 0
    #curr_hist = medfilt(curr_hist, 5)

    if len(final_shifts) >= N:
        #DOUBT: Mean of predicted shifts or actual shifts?
        pred_shifts.append(int(np.mean(final_shifts)+0.5))
        shift_thresh = int(np.sqrt(pred_shifts[-1]))
        w = pred_shifts[-1]

    min_corr = 1e5
    for j in range(-w, w):
        shift = pred_shifts[-1] + j
        start_pt = max(0,start[-1]+shift)
        end_pt = min(union_length, end[-1]+shift)
        prev_hist = hist_union[start_pt:end_pt]
        if start_pt == 0 and len(prev_hist) != hist_length:
            corr = sim_measure(curr_hist[-len(prev_hist):], prev_hist, i, j+100)
        elif end_pt == union_length and len(prev_hist) != hist_length:
            corr = sim_measure(curr_hist[:len(prev_hist)], prev_hist, i, j+100)
        else:
            corr = sim_measure(curr_hist, prev_hist, i, j+100)
        if corr <= min_corr and corr != 0:
            min_corr = corr
            final_shift = shift

    #TEMP FIX for 0 final shifts when predictions are not on
    if final_shift <= 0 and len(final_shifts) < N:
        final_shift = final_shifts[-1]

    if len(pred_shifts) > 1:
        if abs(final_shift - pred_shifts[-1]) > shift_thresh:
            final_shift = pred_shifts[-1]
    final_shifts.append(final_shift)

    #Update start and end to get correct prev_hist every iteration
    start.append(start[-1] + final_shift)
    end.append(end[-1] + final_shift)

    hist_union[start[-1]:end[-1]] += curr_hist
    hist_union[start[-1]:end[-2]] /= 2

    if i == N:
        break

str_version = 'N-{}_w-{}_it-{}_st-{}_sw-{}_la-{}_d-{}'\
                 .format(N,w,intensity_thresh,shift_thresh,smooth_window, lookahead, delta)
end_time = time.time()
dur3 = end_time - start_time
print 'Duration: {}'.format(dur3)
print 'DONE'
###############################################################################

#Part 5: Finding minimas ######################################################
start_time = time.time()
car_lengths = []
print 'Finding minimas ... '
gnd_truth = pd.read_csv(stem + 'ground_truth.csv')
smooth_union = medfilt(hist_union, smooth_window)
_, _min = peakdetect(smooth_union,lookahead = lookahead, delta = delta)
pred_gaps = np.asarray([p[0] for p in _min])
yn = np.asarray([p[1] for p in _min])
pred_gaps = np.insert(pred_gaps, 0, np.where(smooth_union != 0)[0][0])
pred_gaps = np.insert(pred_gaps, len(pred_gaps), np.where(smooth_union != 0)[0][-1])
yn = np.insert(yn, 0, 0)
yn = np.insert(yn, len(yn), 0)

initial_pred_gaps = pred_gaps
#Finding missed out gaps
true_lengths = gnd_truth['Train lengths']
thresh = 1.1 * max(true_lengths)
#thresh = 1920
temp = []

temp_start = []
for i in range(len(pred_gaps)-1):
    if pred_gaps[i+1] - pred_gaps[i] > thresh:
        temp.append(smooth_union[pred_gaps[i]:pred_gaps[i+1]])
        temp_start.append(pred_gaps[i])


for i in range(len(temp)):
    temp_inv = np.asarray([abs(t - max(temp[i])) for t in temp[i]])
    temp_window =  int(len(temp_inv)*0.0025)
    if temp_window % 2 == 0:
        temp_window += 1
    temp_inv = medfilt(temp_inv, temp_window)
        
    pos_pts = np.where(temp_inv != 0)[0]
    
    groups = grouper(pos_pts)
    group_vals = [temp_inv[g].tolist() for g in groups]
    
    group_cents = []
    for k in range(len(groups)):
        group_cents.append(groups[k][group_vals[k].index(max(group_vals[k]))])
    
    final_mins = [group_cents[0]]
    final_abs_mins = []
    final_yn = []
    j = 1
    while j < len(group_cents):
        if group_cents[j] - final_mins[-1] > 0.9 * min(true_lengths): #750
            final_mins.append(group_cents[j])
            final_abs_mins.append(temp_start[i] + group_cents[j])
            final_yn.append(smooth_union[final_abs_mins[-1]])
            j += 1
        else:
            j += 1
            
    pred_gaps = np.append(pred_gaps, np.asarray(final_abs_mins[:-1]))
    yn = np.append(yn, np.asarray(final_yn[:-1]))

yn = [x for (y,x) in sorted(zip(pred_gaps,yn))]
pred_gaps = np.sort(pred_gaps)

#Removing close gaps overall
final_pred_gaps = [pred_gaps[0]]
final_yn = [yn[0]]
i = 1
while i < len(pred_gaps):
    if pred_gaps[i] - final_pred_gaps[-1] > 0.9 * min(true_lengths): #750
        final_pred_gaps.append(pred_gaps[i])
        final_yn.append(yn[i])
        i = i+1
    else:
        i = i+1

pred_gaps = np.asarray(final_pred_gaps, dtype=int)
yn = np.asarray(final_yn, dtype=int)

pt_no = np.linspace(0,len(pred_gaps)-1, len(pred_gaps)).astype(np.uint8)
fig = plt.figure(figsize=(20,4))
plt.xticks(np.arange(0, len(smooth_union)+1, 1000))
ax = fig.add_subplot(111)
ax.grid()
plt.plot(smooth_union)
plt.scatter(pred_gaps, yn, color = 'red')
for i, txt in enumerate(pt_no):
    ax.annotate(txt, (pred_gaps[i],max(0, yn[i])))
plt.savefig(graph_dir +  stem[:-1] + '_preds_' + str_version + '.jpg')
plt.close()

for i in range(len(pred_gaps)-1):
    diff = pred_gaps[i+1] - pred_gaps[i]
    car_lengths.append(diff)

end_time = time.time()
dur4 = end_time - start_time
print 'Duration: {}'.format(dur4)
print 'DONE'
###############################################################################

#Part 6: Cropping #############################################################
print 'Generating crops ... '
start_time = time.time()
req_frames = []
req_start = []
req_end = []
car_lengths = []
#Left to right
if direc.startswith('L'):
    for i in range(1,len(pred_gaps)):
        for j in range(len(end)):
            if end[j] > pred_gaps[i]:
                req_frames.append(fnums[j])
                req_start.append(end[j] - pred_gaps[i])
                car_lengths.append(pred_gaps[i] - pred_gaps[i-1])
                req_end.append(req_start[-1] + car_lengths[-1])
                break

#Right to left
elif direc.startswith('R'):
    for i in range(1,len(pred_gaps)):
        for j in range(len(end)):
            if end[j] > pred_gaps[i]:
                req_frames.append(fnums[j])
                req_end.append(n - (end[j] - pred_gaps[i]))
                car_lengths.append(pred_gaps[i] - pred_gaps[i-1])
                req_start.append(req_end[-1] - car_lengths[-1])
                break

for i in range(len(req_frames)):
    im = cv2.imread(stem + 'frames/frame' + str(req_frames[i]) + '.jpg')
    crop_im = im[horz_bndry_1:horz_bndry_2,max(0, req_start[i]-10):req_end[i]+10,:]
    cv2.imwrite(stem + 'crops/car_' + str(i) + '.jpg', crop_im)
end_time = time.time()
print 'DONE'
dur5 = end_time - start_time
print 'Duration: {}'.format(dur5)
###############################################################################

#Part 7: Ground truth analysis ################################################
initial_error_1 = []
initial_error_2 = []
error_1 = []
error_2 = []

start_time = time.time()
plt.figure(figsize=(20,4))
plt.plot(smooth_union)
plt.scatter(pred_gaps, yn, color = 'red')

#for i,v in gnd_truth.iterrows():
#    plt.axvline(x=int(v['Gaps']), linestyle='--', color='g')

global_start = [np.where(smooth_union != 0)[0][0]]
plt.axvline(x=int(global_start[0]), linestyle='--', color='g')
for i,v in gnd_truth.iterrows():
    if math.isnan(v['Frame start']) == True:
        break
    start_pt = max(0, int(v['Frame start']))
    global_start.append(start[fnums.index(int(v['Frame number']))] - start_pt)
    plt.axvline(x=int(global_start[-1]), linestyle='--', color='g')

plt.savefig(graph_dir +  stem[:-1] + '_gnd_truth_' + str_version + '.jpg')
plt.close()

global_gaps = gnd_truth['Gaps']
for pg in initial_pred_gaps:
    min_dist = 1e5
    for gs in global_gaps:
        if abs(gs - pg) < min_dist:
            min_dist = abs(gs - pg)
    initial_error_1.append(min_dist)

for gs in global_gaps:
    min_dist = 1e5
    for pg in initial_pred_gaps:  
        if abs(gs - pg) < min_dist:
            min_dist = abs(gs - pg)
    initial_error_2.append(min_dist)

for pg in pred_gaps:
    min_dist = 1e5
    for gs in global_gaps:
        if abs(gs - pg) < min_dist:
            min_dist = abs(gs - pg)
    error_1.append(min_dist)

for gs in global_gaps:
    min_dist = 1e5
    for pg in pred_gaps:  
        if abs(gs - pg) < min_dist:
            min_dist = abs(gs - pg)
    error_2.append(min_dist)

end_time = time.time()
print 'DONE'
dur6 = end_time - start_time
print 'Duration: {}'.format(dur6)
###############################################################################

#Part 4: Graphs ###############################################################
fig = plt.figure(figsize=(20, 4))
ax = fig.add_subplot(111)
ax.grid()
plt.figure(figsize=(20,4))
plt.plot(hist_union)
plt.savefig(graph_dir + stem[:-1] + '_union_hist_' + str_version + '.jpg')
plt.close()

fig = plt.figure(figsize=(20, 4))
ax = fig.add_subplot(111)
ax.grid()
plt.plot(final_shifts)
plt.plot(pred_shifts)
plt.savefig(graph_dir +  stem[:-1] + '_shifts_' + str_version + '.jpg')
plt.close()

fig = plt.figure(figsize=(20, 4))
ax = fig.add_subplot(111)
ax.grid()
plt.plot(abs(np.asarray(final_shifts[-len(pred_shifts):]) - np.asarray(pred_shifts)))
plt.savefig(graph_dir +  stem[:-1] + '_diff_' + str_version + '.jpg')
plt.close()

fig = plt.figure(figsize=(20,4))
ax = fig.add_subplot(111)
ax.grid()
plt.xticks(np.arange(0, max(car_lengths)/100 * 100 + 100, 50))
plt.hist(car_lengths, bins=np.arange(min(car_lengths)/100 * 100, max(car_lengths)/100 * 100 + 100, 100))
plt.xlabel('Car length')
plt.ylabel('Number of cars')
plt.savefig(graph_dir + stem[:-1] + '_car_lengths_' + str_version + '.jpg')
plt.close()

fig = plt.figure(figsize=(20,4))
ax = fig.add_subplot(111)
ax.grid()
plt.xticks(np.arange(0, max(error_1)+100, 100))
plt.hist(error_1, bins=np.arange(0, max(error_1)+100, 100), alpha=0.5, color='g')
plt.xlabel('Minimum error')
plt.ylabel('Number of points in this error range')
plt.savefig(graph_dir + stem[:-1] + '_error1_' + str_version + '.jpg')
plt.close()

fig = plt.figure(figsize=(20,4))
ax = fig.add_subplot(111)
ax.grid()
plt.xticks(np.arange(0, max(error_2)+100, 100))
plt.hist(error_2, bins=np.arange(0, max(error_2)+100, 100), alpha=0.5, color='b')
plt.xlabel('Minimum error')
plt.ylabel('Number of points in this error range')
plt.savefig(graph_dir + stem[:-1] + '_error2_' + str_version + '.jpg')
plt.close()
###############################################################################

#Part 8: Generate HTML ########################################################
f = open(stem + 'video_summary_' + str_version + '.html', 'w')

f.write('<h1>General information</h1>')
write_str = '<p><b>Video name:</b> {0}</p>\
        <p><b>Video link:</b> {1}</p>\
        <p><b>Resolution:</b> {2}x{3}</p>\
        <p><b>Number of frames:</b> {4}</p>'.format(video_name, video_link, m, n, len(fnums))
f.write(write_str)

write_str = '<p><b>Number of cars:</b> {0}</p>\
            <p><b>Average speed:</b> {1} pixels/frame</p>'.format(len(pred_gaps)-1, np.mean(final_shifts))
f.write(write_str)

f.write('<h1>Time Profile</h1>')
time_str = '<p>Time elapsed in function find_horz_boundaries is: {} s</p>'.format(dur1)
time_str += '<p>Time elapsed in function find_direc is: {} s</p>'.format(dur2)
time_str += '<p>Time elapsed in function align is: {} s</p>'.format(dur3)
time_str += '<p>Time elapsed in function find_gap_pos is: {} s</p>'.format(dur4)
time_str += '<p>Time elapsed in function generate_crops is: {} s</p>'.format(dur5)
time_str += '<p>Time elapsed in function ground_truth_analysis is: {} s</p>'.format(dur6)
total_time = dur1 + dur2 + dur3 + dur4 + dur5 + dur6
time_str += '<p><b>Total duration is:</b> {} s</p>'.format(total_time)
f.write(time_str)

f.write('<h1>Graphs</h1>')
fname = graph_dir + stem[:-1] + '_union_hist_' + str_version + '.jpg'
data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
img_tag = '<table class="image"><caption align="bottom">Union Histogram</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
f.write(img_tag)


fname = graph_dir + stem[:-1] + '_shifts_' + str_version + '.jpg'
data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
img_tag = '<table class="image"><caption align="bottom">Speed Graph</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
f.write(img_tag)

fname = graph_dir + stem[:-1] + '_diff_' + str_version + '.jpg'
data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
img_tag = '<table class="image"><caption align="bottom">Offset between predicted and actual shifts</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
f.write(img_tag)

fname = graph_dir + stem[:-1] + '_preds_' + str_version + '.jpg'
data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
img_tag = '<table class="image"><caption align="bottom">Histogram union with predicted gap positions</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
f.write(img_tag)

fname = graph_dir + stem[:-1] + '_car_lengths_' + str_version + '.jpg'
data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
img_tag = '<table class="image"><caption align="bottom">Car length histogram</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
f.write(img_tag)

fname = graph_dir + stem[:-1] + '_gnd_truth_' + str_version + '.jpg'
data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
img_tag = '<table class="image"><caption align="bottom">Position of ground truth gaps on union histogram</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
f.write(img_tag)

fname = graph_dir + stem[:-1] + '_error1_' + str_version + '.jpg'
data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
img_tag = '<table class="image"><caption align="bottom">Error between predicted cuts and ground truth</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
f.write(img_tag)        
f.write('<p>INITIAL STAGE | L2 distance between real and closest cuts  (min, mean and max): {}, {}, {}</p>'\
       .format(min(initial_error_1), np.mean(initial_error_1), max(initial_error_1)))
f.write('<p>FINAL STAGE | L2 distance between real and closest cuts  (min, mean and max): {}, {}, {}</p>'\
       .format(min(error_1), np.mean(error_1), max(error_1)))

fname = graph_dir + stem[:-1] + '_error2_' + str_version + '.jpg'
data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
img_tag = '<table class="image"><caption align="bottom">Error between ground truth and predicted cuts</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
f.write(img_tag)
f.write('<p>INITIAL STAGE | L2 distance between cuts and closet reals  (min, mean and max): {}, {}, {}</p>'\
       .format(min(initial_error_2), np.mean(initial_error_2), max(initial_error_2)))
f.write('<p>FINAL STAGE | L2 distance between cuts and closet reals  (min, mean and max): {}, {}, {}</p>'\
       .format(min(error_2), np.mean(error_2), max(error_2)))

f.write('<h1>Crops</h1>')
for i in range(len(req_frames)):
    fname = stem + 'crops/car_' + str(i) + '.jpg'
    data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
    img_tag = '<figure><img src="data:image/png;base64,{0}">\
    <figcaption>Car #{1}, Frame:{2}, Length: {3}</figcaption></figure>'\
                                  .format(data_uri, i, req_frames[i], car_lengths[i])
    f.write(img_tag)

f.close()
###############################################################################