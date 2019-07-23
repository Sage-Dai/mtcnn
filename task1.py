# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 20:31:39 2019

@author: 我的天呐
"""

import os
name = input('your name:')
if os.path.exists(name)!=True:
    os.mkdir(name)
    
import cv2
cap = cv2.VideoCature(0)

flag = 1
num = 1
while(cap.isOpened()):
    ret_flag, frame = cap.read()
    cv2.imshow('photos',frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        cv2.imwrite('./faceImage/{}.jpg'.format(num),frame)
        print('success to save'+str(num)+'.jpg')
        print('------------')
        num += 1
    elif k == 27:
        break
cap.release()
cv2.destroyAllWindows()