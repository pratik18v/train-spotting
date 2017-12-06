import os
import cv2

factor = 1
for vid_name in os.listdir('videos'):
  if 'DS_Store' in vid_name: 
    continue
  stem = vid_name.split('.')[0]
  vidcap = cv2.VideoCapture('videos/' + vid_name)
  #vidcap = cv2.VideoCapture('BB_NS_mixed_freight.MOV')
  success,image = vidcap.read()
  rows, cols = 720, 1280
  i = 0
  count = 0
  success = True
  M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
  
  if os.path.exists(stem + '/') == False:
      os.mkdir(stem + '/')
  if os.path.exists(stem + '/frames/') == False:
      os.mkdir(stem + '/frames/')
  
  while success:
      success, image = vidcap.read()
      if i%factor == 0:
          #print stem + "/frames_1920_fps" + str(30/factor) + "/frame%d.jpg" % i
          #cv2.imwrite(stem + "/frames_1920_fps" + str(30/factor) + "/frame%d.jpg" % i, image)     # save frame as JPEG file
          #cv2.imwrite(stem + "/frames/image_%d.jpg" % count, image)     # save frame as JPEG file
          image = cv2.warpAffine(image,M,(cols,rows))
          cv2.imwrite(stem + "/frames/image_%d.jpg" % count, image)     # save frame as JPEG file
          count += 1
      i += 1        
  print 'Total frames written: {}'.format(count)