#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:27:12 2017

@author: pratik18v
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage.filters import convolve
from shutil import copyfile

start_frame = 225.0
end_frame = 241.0

name = 'portjeff_11'
folder = name + '/frames/'
scores = []

num_files = len(os.listdir(folder))
img_ext = 'image_'
num_bkg_imgs = 10
bs_thresh = 20
bkg_img = cv2.imread(folder + img_ext + str(1) + '.jpg', 0).astype(np.float64)
m,n = bkg_img.shape

for i in range(2, num_bkg_imgs):
  bkg_img += cv2.imread(folder + img_ext + str(i) + '.jpg', 0).astype(np.float64)
  
bkg_img /= num_bkg_imgs
flag = False
for i in range(num_bkg_imgs, num_files):
  im = cv2.imread(folder + img_ext + str(i) + '.jpg', 0).astype(np.float64)
  diff = bkg_img - im
  diff = cv2.GaussianBlur(diff,(11,11),0)
  th1 = (diff > bs_thresh) * 255
  th1[:450,:] = 0
  th1[600:,:] = 0
  th1[:,600:] = 0
  #cv2.imwrite('temp/' + str(i) +'.jpg', th1)
  score = 1.0 * (np.where(th1 == 255)[0].shape[0]) / (m*n)
  if score < 0.02 and flag == False:
    bkg_img += im
    bkg_img /= 2
  else:
    flag = True
  scores.append(score)
#  cv2.imwrite('temp/' + str(i) + '.jpg', th1)
  
if os.path.exists(name + '/results') == False:
      os.mkdir(name + '/results')
      
smooth_scores = medfilt(scores, 11)
plt.figure(figsize=(10,5))
plt.ylabel('Similarity score')
plt.xlabel('Time passed (in sec)')
plt.plot(smooth_scores)
plt.savefig(name + '/results/scores.jpg')
plt.close()

indicator = convolve(smooth_scores, [-1,1])
plt.figure(figsize=(10,5))
plt.ylabel('Relative change in score')
plt.xlabel('Time passed (in sec)')
plt.plot(indicator)
plt.savefig(name + '/results/indicator.jpg')
plt.close()

max_val = np.max(indicator)
min_val = np.min(indicator)

max_ind = (indicator / max_val) * 100
min_ind = (indicator / min_val) * 100

start_k = 1.0 * (num_bkg_imgs + np.where(min_ind == 100)[0][0])
end_k = 1.0 * (num_bkg_imgs + np.where(max_ind == 100)[0][0])

print 'Train starts at: {}'.format(start_k)
print 'Train ends at: {}'.format(end_k)

# stripped
if os.path.exists(name + '/stripped') == False:
      os.mkdir(name + '/stripped')
      
outdir = name + '/stripped'     
startin = int(start_k)
endin = int(end_k)
#startin = 155
#endin = 170
count = 0
for filename in os.listdir(folder):
    index = int(filename.split('_')[1].split('.')[0])
    if index >= startin and index <= endin:
        srcfile = os.path.join(folder, filename)
        dstfile = os.path.join(outdir, 'image_%d.jpg' %count)
        copyfile(srcfile, dstfile)
        count += 1

overlap = min(end_frame, end_k) - max(start_frame, start_k) + 1
overlap = max(0, overlap)
union = (end_frame - start_frame + 1) + (end_k - start_k + 1) - overlap
tiou = overlap / union

print 'temporal IoU: {}'.format(tiou)