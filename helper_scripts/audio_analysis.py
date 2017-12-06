#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 03:19:46 2017

@author: pratik18v
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt


def cluster(data, maxgap):
    '''Arrange data into groups where successive elements
       differ by no more than *maxgap*

        >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
        [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

        >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
        [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]

    '''
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups

stem = 'bb_19'
X,sr = librosa.load('/Users/pratik18v/Documents/train-spotting/audio/' + stem + '_full_audio.m4a')
X[X > 0.5  * np.max(X)] = 1
idxs = np.where(X == 1)[0]
idxs /= 21600
groups = cluster(idxs, 5)
width = [len(g) for g in groups]
start_k = groups[width.index(max(width))][0] #/ 21600
end_k = groups[width.index(max(width))][-1] #/ 21600
     
frame_no = np.linspace(0, X.shape[0]-1, X.shape[0])
frame_no /= 21600
plt.figure(figsize=(10,4))
plt.plot(frame_no, X)
if end_k - start_k < 5:
  print 'No train'

plt.title(str(start_k) + ', ' + str(end_k))
plt.savefig('audio/' + stem + '.jpg')

#bb2: 0:00 - 0:04
#bb6: 2:00 - 2:13
#bb7: 2:18 - 2:26
