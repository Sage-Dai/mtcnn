# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:22:36 2019
@author: 我
"""
import tensorflow as tf 
import detect_face 
import cv2 
import re,os
##获取所有文件路径
path = './faceImages/ljh/'
def getimgnames(path=path):
    filenames = os.listdir(path)
    imgnames = []
    for i in filenames:
        if re.findall('^\d+\.jpg$',i)!=[]:
            imgnames.append(i)
    return imgnames
imgnames = getimgnames(path)
##参数设置
minsize = 25 # 脸矩阵最小值 
threshold = [ 0.7, 0.8, 0.7 ] # 三步参数 
factor = 0.609 # 过滤因子
gpu_memory_fraction=1.0

##创建网络参数
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction) 
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) 
    with sess.as_default(): 
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None) 
        
   #读取图片
num=0
for image_path in imgnames:
    img = cv2.imread(path+image_path)
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor) 
    if bounding_boxes.shape[0]:
        face_position=bounding_boxes[0]
        face_position=face_position.astype(int) 
        cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2) 
        crop=img[face_position[1]:face_position[3], 
             face_position[0]:face_position[2],] 
        gary = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        cv2.imwrite('./faceImages/ljhtest/{}.jpg'.format(num), gary)
        num = num+1
    else:
        print('第',num,'张图无法识别')
        num = num+1

