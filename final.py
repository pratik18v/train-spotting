#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 13:26:57 2017

@author: pratik18v
"""

import cv2
import glob
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from peakdetect import peakdetect

class video_analysis(object):
    
    def __init__(self, main_folder, sub_folder1, sub_folder2, sub_folder3,\
                 sub_folder4, sub_folder5, hist_length, shift_threshold, \
                 N=5, w=50, smooth_window=15, lookahead=10, delta=5):

        self.main_folder = main_folder
        self.sub_folder1 = sub_folder1
        self.sub_folder2 = sub_folder2
        self.sub_folder3 = sub_folder3
        self.sub_folder4 = sub_folder4
        self.sub_folder5 = sub_folder5
        self.fnums = []
        self.num_files = 0
        self.N = N
        self.w = w
        self.hist_length = hist_length
        self.rows = [np.zeros([self.hist_length])]
        self.hists = []
        self.shift_threshold = shift_threshold
        self.final_shifts = []
        self.final_corrs = []
        self.act_rel_shifts = []
        self.pred_abs_shifts = []
        self.pred_rel_shifts = []
        self.smooth_window = smooth_window
        self.lookahead = lookahead
        self.delta = delta
        self.car_lengths = []
        self.error_1 = []
        self.error_2 = []
        self.str_version = 'N-{}_w-{}_shift_tresh-{}'.format(self.N, self.w, self.shift_threshold)
#        self.bndry1
#        self.bndry2
#        self.m
#        self.n
#        self.union_norm
#        self.pred_gaps
#        self.yn

    def sim_measure(self, vec1, vec2):
        return pow(sum(pow((vec1 - vec2), 2)), 0.5)/ len(vec1)

    #folder = 'train_crossing_3/opt_flow_3/full_res/'
    def find_horz_boundaries(self):
        im = cv2.imread(self.main_folder + self.sub_folder1 + 'opticalhsv0.png', 0)
        [self.m,self.n] = im.shape
        avg_im = np.zeros([self.m,self.n])
        ctr = 0
        for fname in glob.glob(self.main_folder + self.sub_folder1+'*.png'):
            avg_im += cv2.imread(fname, 0)
            ctr += 1
        avg_im = avg_im/ctr
        
        ret2,th2 = cv2.threshold(np.asarray(avg_im, dtype=np.uint8),0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(th2,kernel,iterations = 2)
        dilation = cv2.dilate(erosion,kernel,iterations = 8)
    
        row_sums = np.sum(dilation, axis=1).tolist()
        for rs in row_sums:
            if int(rs) > 0.5*self.n:
                self.bndry_1 = row_sums.index(rs)
                break
                
        for rs in row_sums[::-1]:
            if int(rs) > 0.5*self.n:
                self.bndry_2 = row_sums.index(rs)
                break

    #folder = 'train_crossing_3/blob_output/full_res/'
    def gen_file_list(self):
        for fname in glob.glob(self.main_folder + self.sub_folder2+'*.jpg'):
            self.fnums.append(int(fname.split('.')[0].split('frame')[-1]))
            self.num_files += 1
    
        self.fnums.sort()

    def align(self):
        pred_shift = 0
        #self.hist_length = self.n
        #self.rows = [np.zeros([self.hist_length])]
        for i in self.fnums:
            print 'Frames processed: {}'.format(i)
        
            aligned = False
            #Get blob image
            bkg = cv2.imread(self.main_folder + self.sub_folder2 + 'frame' + str(i) + '.jpg', 0)

            #Generate histogram of white pixels (first hist_length pixels)
            bkg = bkg[self.bndry_1:self.bndry_2,:]
            row_thresh = np.sum(bkg, axis=0) / 255
            #row_thresh[800:875] = row_thresh[799]
            row_thresh = row_thresh[:self.hist_length]
            row_thresh = row_thresh[::-1]
            #TEMP BUG FIX: last element always zero
            row_thresh[-1] = row_thresh[-2]
            
            row_thresh[row_thresh[:] > 20] = 20

            if len(self.hists) >= 1:
                prev_hist = self.hists[-1]
                curr_hist = row_thresh

                if len(self.final_shifts) == 1:
                    self.act_rel_shifts.append(self.final_shifts[-1])
                elif len(self.final_shifts) > 1 and len(self.final_shifts) <= self.N:
                    self.act_rel_shifts.append(self.final_shifts[-1] - self.final_shifts[-2])
                elif len(self.final_shifts) >self.N:
                    rel_shift = int(np.mean(self.act_rel_shifts)+0.5)
                    pred_shift = (len(prev_hist) - self.hist_length) + rel_shift

                    self.pred_rel_shifts.append(rel_shift)
                    self.pred_abs_shifts.append(pred_shift)

                corrs = []
                shifts = []
                for j in range(-self.w, self.w+1):
                    shift = pred_shift + j
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

                    corr = self.sim_measure(prev_aligned_hist, curr_aligned_hist)
                    corrs.append(corr)
                    shifts.append(shift)

                final_shift = shifts[corrs.index(min(corrs))]
                self.final_corrs.append(min(corrs))

                if len(self.final_shifts) > 0:
                    if final_shift - self.final_shifts[-1] <= self.shift_threshold:
                        final_shift = pred_shift

                self.final_shifts.append(final_shift)
        
                if self.final_shifts[-1] > 0:
                    padding = np.zeros([self.final_shifts[-1]])
                    max_aligned = np.hstack((padding, curr_hist))
                else:
                    curr_aligned_hist = curr_hist[abs(self.final_shifts[-1]):]
                    prev_aligned_hist = prev_hist
                    padding = np.zeros([len(prev_aligned_hist) - len(curr_aligned_hist)])
                    max_aligned = np.hstack((curr_aligned_hist, padding))

                if len(self.final_shifts) > self.N:
                    self.act_rel_shifts.append(self.final_shifts[-1] - self.final_shifts[-2])

                aligned = True
        
            self.rows.append(row_thresh)
            if aligned == True:
                self.hists.append(max_aligned)
            else:
                self.hists.append(row_thresh)

    
    def take_union(self):
        self.start = [0]
        self.end = [self.hist_length]
        for i in range(len(self.final_shifts)):
            next_start = self.final_shifts[i]
            next_end = next_start + self.hist_length
            self.start.append(next_start) 
            self.end.append(next_end)
            
        lens = [len(h) for h in self.hists]
        union = np.zeros([max(lens)])
        divisors = np.zeros([max(lens)])
        
        for i in range(len(self.hists)):
            union[:len(self.hists[i])] += self.hists[i]
        
        for i in range(max(lens)):
            for j in range(len(self.start)):
                if i >= self.start[j] and i <= self.end[j]:
                    divisors[i] += 1
        
        self.union_norm = np.asarray(union) / np.asarray(divisors)

    def find_gap_pos(self):
        self.smooth_union = medfilt(self.union_norm, self.smooth_window)
        _, _min = peakdetect(self.smooth_union,lookahead=self.lookahead, delta=self.delta)
        self.pred_gaps = np.asarray([p[0] for p in _min])
        self.yn = np.asarray([p[1] for p in _min])
        self.pred_gaps = np.insert(self.pred_gaps, 0, np.where(self.smooth_union != 0)[0][0])
        self.pred_gaps = np.insert(self.pred_gaps, len(self.pred_gaps), np.where(self.smooth_union != 0)[0][-1])
        self.yn = np.insert(self.yn, 0, 0)
        self.yn = np.insert(self.yn, len(self.yn), 0)

        for i in range(len(self.pred_gaps)-1):
            diff = self.pred_gaps[i+1] - self.pred_gaps[i]
            self.car_lengths.append(diff)

    def generate_crops(self):
        self.req_frames = []
        self.req_start = []
        self.req_end = []
        self.car_lengths = []
        for i in range(1,len(self.pred_gaps)):
            for j in range(len(self.end)):
                if self.end[j] > self.pred_gaps[i]:
                    self.req_frames.append(self.fnums[j])
                    self.req_start.append(self.end[j] - self.pred_gaps[i])
                    self.car_lengths.append(self.pred_gaps[i] - self.pred_gaps[i-1])
                    self.req_end.append(self.req_start[-1] + self.car_lengths[-1])
                    break
        
        for i in range(len(self.req_frames)):
            #'train_crossing_3/frames/full_res/frame' + str(req_frames[i]) + '.jpg'
            im = cv2.imread(self.main_folder + self.sub_folder3 + 'frame' + str(self.req_frames[i]) + '.jpg')
            crop_im = im[self.bndry_1:self.bndry_2,max(0, self.req_start[i]-10):self.req_end[i]+10,:]
            cv2.imwrite(self.main_folder + self.sub_folder4 + 'car_' + str(i) + '.jpg', crop_im)

    def ground_truth_analysis(self):
        gnd_truth = pd.read_csv(self.main_folder + 'ground_truth.csv')
        factor = 1280.0/480
        self.global_gaps = [np.where(self.smooth_union != 0)[0][0]]
        for i,v in gnd_truth.iterrows():
            if math.isnan(v['Frame start']) == True:
                break
            start_pt = max(0, int(factor*v['Frame start']))
            #end_pt = min(1100, int(factor*v['Frame end']))
            self.global_gaps.append(self.end[self.fnums.index(int(v['Frame number']))] - start_pt)

        for pg in self.pred_gaps:
            min_dist = 1e15
            for gs in self.global_gaps:
                if abs(gs+100 - pg) < min_dist:
                    min_dist = abs(gs+100 - pg)
            self.error_1.append(min_dist)
        
        for gs in self.global_gaps:
            min_dist = 1e15
            for pg in self.pred_gaps:  
                if abs(gs+100 - pg) < min_dist:
                    min_dist = abs(gs+100 - pg)
            self.error_2.append(min_dist)

    def generate_graphs(self):
        #1. Final histogram
        fig = plt.figure(figsize=(20, 4))
        ax = fig.add_subplot(111)
        ax.grid()
        plt.plot(self.union_norm)
        plt.xlabel('row pixel')
        plt.ylabel('avg number of white pixels')
        plt.savefig(self.main_folder + self.sub_folder5 + 'union_hist_' + self.str_version + '.jpg')
        plt.close()
        
        #2. Speed vs frame number
        fig = plt.figure(figsize=(20, 4))
        ax = fig.add_subplot(111)
        ax.grid()
        plt.plot(self.fnums[1:], self.act_rel_shifts, 'bo', markersize=1)
        plt.xlabel('frame number')
        plt.ylabel('speed')
        plt.savefig(self.main_folder + self.sub_folder5 + 'speed_vs_frame_no_' + self.str_version + '.jpg')
        plt.close()
        
        #4. Difference in actual and predicted shifts
        fig = plt.figure(figsize=(20, 4))
        ax = fig.add_subplot(111)
        ax.grid()
        plt.plot(np.asarray(self.final_shifts[-len(self.pred_abs_shifts):]) - np.asarray(self.pred_abs_shifts), 'bo', markersize=1)
        plt.ylabel('Offset')
        plt.xlabel('frame number')
        plt.savefig(self.main_folder + self.sub_folder5 + 'diff_shifts_' + self.str_version + '.jpg')
        plt.close()

        pt_no = np.linspace(0,len(self.pred_gaps)-1, len(self.pred_gaps)).astype(np.uint8)
        fig = plt.figure(figsize=(20,4))
        #plt.xticks(np.arange(0, len(self.smooth_union)+1, 1000))
        ax = fig.add_subplot(111)
        ax.grid()
        plt.plot(self.smooth_union)
        plt.scatter(self.pred_gaps, self.yn, color = 'red')
        for i, txt in enumerate(pt_no):
            ax.annotate(txt, (self.pred_gaps[i],max(0, self.yn[i])))
        plt.savefig(self.main_folder + self.sub_folder5 + 'union_minimas_' + self.str_version + '.jpg')
        plt.close()

        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        ax.grid()
        plt.hist(self.car_lengths, bins=np.arange(min(self.car_lengths)/100 * 100, max(self.car_lengths)/100 * 100 + 100, 50))
        plt.xlabel('Car length')
        plt.ylabel('Number of cars')
        plt.savefig(self.main_folder + self.sub_folder5 + 'car_length_hist_' + self.str_version + '.jpg')
        plt.close()

        fig = plt.figure(figsize=(20,4))
        #plt.xticks(np.arange(0, len(self.smooth_union)+1, 1000))
        ax = fig.add_subplot(111)
        ax.grid()
        plt.plot(self.smooth_union)
        plt.scatter(self.pred_gaps, self.yn, color = 'red')
        for i, txt in enumerate(pt_no):
            ax.annotate(txt, (self.pred_gaps[i],max(0, self.yn[i]-5)))
        #for i,v in gnd_truth.iterrows():
        for i in range(len(self.global_gaps)):
            #plt.axvline(x=int(factor*v['Gaps']), linestyle='--', color='g')
            plt.axvline(x=self.global_gaps[i], linestyle='--', color='m')
            #plt.axvline(x=global_end[i], linestyle='--', color='m')
        plt.savefig(self.main_folder + self.sub_folder5 + 'ground_truth_' + self.str_version + '.jpg')
        plt.close()
        
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        ax.grid()
        plt.xticks(np.arange(0, max(self.error_1)+10, 10))
        plt.hist(self.error_1, bins=np.arange(0, max(self.error_1)+10, 10), alpha=0.5, color='g')
        plt.xlabel('Minimum error')
        plt.ylabel('Number of points in this error range')
        plt.legend(['Error between predicted cuts and ground truth', 'Error between ground truth and predicted cuts'])
        plt.savefig(self.main_folder + self.sub_folder5 + 'error1_' + self.str_version + '.jpg')
        plt.close()

        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        ax.grid()
        plt.xticks(np.arange(0, max(self.error_2)+10, 10))
        plt.hist(self.error_2, bins=np.arange(0, max(self.error_2)+10, 10), alpha=0.5, color='b')
        plt.xlabel('Minimum error')
        plt.ylabel('Number of points in this error range')
        plt.legend(['Error between predicted cuts and ground truth', 'Error between ground truth and predicted cuts'])
        plt.savefig(self.main_folder + self.sub_folder5 + 'error2_' + self.str_version + '.jpg')
        plt.close()

    def generate_html(self, time_str):
        video_name = 'train_crossing_3.mp4'
        
        f = open(self.main_folder + 'video_summary_' + self.str_version + '.html', 'w')
    
        f.write('<h1>General information</h1>')
        write_str = '<p><b>Video name:</b> {0}</p>\
                <p><b>Resolution:</b> {1}x{2}</p>\
                <p><b>Number of frames:</b> {3}</p>'.format(video_name, self.m, self.n, self.num_files)
        f.write(write_str)
        
        write_str = '<p><b>Number of cars:</b> {0}</p>\
                    <p><b>Average speed:</b> {1} pixels/frame</p>'.format(len(self.pred_gaps)-1, np.mean(self.act_rel_shifts))
        f.write(write_str)

        f.write('<h1>Time Profile</h1>')
        f.write(time_str)

        f.write('<h1>Graphs</h1>')
        fname = self.main_folder + self.sub_folder5 + 'union_hist_' + self.str_version + '.jpg'
        data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
        img_tag = '<table class="image"><caption align="bottom">Union Histogram</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
        f.write(img_tag)
        
    
        fname = self.main_folder + self.sub_folder5 + 'speed_vs_frame_no_' + self.str_version + '.jpg'
        data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
        img_tag = '<table class="image"><caption align="bottom">Seepd Graph</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
        f.write(img_tag)
        
        fname = self.main_folder + self.sub_folder5 + 'diff_shifts_' + self.str_version + '.jpg'
        data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
        img_tag = '<table class="image"><caption align="bottom">Offset between predicted and actual shifts</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
        f.write(img_tag)
        
        fname = self.main_folder + self.sub_folder5 + 'union_minimas_' + self.str_version + '.jpg'
        data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
        img_tag = '<table class="image"><caption align="bottom">Histogram union with predicted gap positions</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
        f.write(img_tag)

        fname = self.main_folder + self.sub_folder5 + 'car_length_hist_' + self.str_version + '.jpg'
        data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
        img_tag = '<table class="image"><caption align="bottom">Car length histogram</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
        f.write(img_tag)

        fname = self.main_folder + self.sub_folder5 + 'ground_truth_' + self.str_version + '.jpg'
        data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
        img_tag = '<table class="image"><caption align="bottom">Car length histogram</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
        f.write(img_tag)

        fname = self.main_folder + self.sub_folder5 + 'error1_' + self.str_version + '.jpg'
        data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
        img_tag = '<table class="image"><caption align="bottom">Error between predicted cuts and ground truth</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
        f.write(img_tag)        
        f.write('<p>L2 distance between real and closest cuts  (mean and max): {}, {}</p>'\
               .format(np.mean(self.error_1), max(self.error_1)))

        fname = self.main_folder + self.sub_folder5 + 'error2_' + self.str_version + '.jpg'
        data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
        img_tag = '<table class="image"><caption align="bottom">Error between ground truth and predicted cuts</caption><tr><td><img src="data:image/png;base64,{0}"></td></tr></table>'.format(data_uri)
        f.write(img_tag)
        f.write('<p>L2 distance between cuts and closet reals  (mean and max): {}, {}</p>'\
               .format(np.mean(self.error_2), max(self.error_2)))

        f.write('<h1>Crops</h1>')
        for i in range(len(self.req_frames)):
            fname = self.main_folder + self.sub_folder4 + 'car_' + str(i) + '.jpg'
            data_uri = open(fname, 'rb').read().encode('base64').replace('\n', '')
            img_tag = '<figure><img src="data:image/png;base64,{0}">\
            <figcaption>Car #{1}, Frame:{2}, Length: {3}</figcaption></figure>'\
                                          .format(data_uri, i, self.req_frames[i], self.car_lengths[i])
            f.write(img_tag)
        
        f.close()
        
if __name__ == '__main__':
    
    main_folder = '/Users/pratik18v/Documents/train-spotting/train_crossing_3/'
    sub_folder1 = 'opt_flow/full_res/'
    sub_folder2 = 'blob_output/full_res/'
    sub_folder3 = 'frames/full_res/'
    sub_folder4 = 'crops/full_res/'
    sub_folder5 = 'graphs/full_res/'
    hist_length = 800
    shift_threshold = 17
    
    algo = video_analysis(main_folder, sub_folder1, sub_folder2, sub_folder3, \
                          sub_folder4, sub_folder5, hist_length, shift_threshold)
    
    start_time = time.time()
    algo.find_horz_boundaries()
    end_time = time.time()
    dur1 = end_time - start_time
    time_str1 = '<p>Time elapsed in function find_horz_boundaries is: {}</p> sec'.format(dur1)
    print time_str1
    
    start_time = time.time()
    algo.gen_file_list()
    end_time = time.time()
    dur2 = end_time - start_time
    time_str2 = '<p>Time elapsed in function gen_file_list is: {}</p> sec'.format(dur2)
    print time_str2

    start_time = time.time()
    algo.align()
    end_time = time.time()
    dur3 = end_time - start_time
    time_str3 = '<p>Time elapsed in function align is: {}</p> s'.format(dur3)
    print time_str3

    start_time = time.time()
    algo.take_union()
    end_time = time.time()
    dur4 = end_time - start_time
    time_str4 = '<p>Time elapsed in function take_union is: {}</p> s'.format(dur4)
    print time_str4

    start_time = time.time()
    algo.find_gap_pos()
    end_time = time.time()
    dur5 = end_time - start_time
    time_str5 = '<p>Time elapsed in function find_gap_pos is: {}</p> s'.format(dur5)
    print time_str5

    start_time = time.time()
    algo.generate_crops()
    end_time = time.time()
    dur6 = end_time - start_time
    time_str6 = '<p>Time elapsed in function generate_crops is: {}</p> s'.format(dur6)
    print time_str6

    start_time = time.time()
    algo.ground_truth_analysis()
    end_time = time.time()
    dur7 = end_time - start_time
    time_str7 = '<p>Time elapsed in function ground_truth_analysis is: {} s</p>'.format(dur7)
    print time_str7

    start_time = time.time()
    algo.generate_graphs()
    end_time = time.time()
    dur8 = end_time - start_time
    time_str8 = '<p>Time elapsed in function generate_graphs is: {} s</p>'.format(dur7)
    print time_str8

    total_time = dur1 + dur2 + dur3 + dur4 + dur5 + dur6 + dur7 + dur8
    time_str9 = '<p><b>Total duration is:</b> {} s</p>'.format(total_time)
    algo.generate_html(time_str1 + time_str2 + time_str3 + time_str4 + time_str5 + time_str6 + time_str7 + time_str8 + time_str9)
