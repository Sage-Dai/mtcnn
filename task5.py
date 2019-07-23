import cv2
import tensorflow as tf
import detect_face
import numpy as np
#import os
#参数设置
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
gpu_memory_fraction=1.0
Name = ['daiyejun','gaohongbin','heziwei','lgh','ljh','lzy','pcm','renhuikang','wangjianwei','wy']
#filenames = os.listdir('E:/spyder/faceImageGray')
#dic = {}
#for i in range(10):
#    dic[filenames[i]] = i
#    dic[i] = filenames[i]

#for i in range(10):
#    name_labels[i] = filenames[i]
#    onehot_labels[i] = tf.one_hot(filenames[i],10)
#labels = np.vstack((name_labels,onehot_labels))

#detect_face模型调用
with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
            
sess = tf.Session()
saver = tf.train.import_meta_graph('E:/spyder/cnn/model_1/cnnmodel.meta')
saver.restore(sess,tf.train.latest_checkpoint('E:/spyder/cnn/model_1'))
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret,frame = cap.read()
    #获取人脸数据
    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
    if bounding_boxes.shape[0]!=0:
        face_position = bounding_boxes[0]
        face_position = face_position.astype(int)
        cv2.rectangle(frame, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (255, 0, 0), 2)
        img_new = frame[face_position[1]:face_position[3],face_position[0]:face_position[2]]
        img_new = cv2.cvtColor(img_new,cv2.COLOR_BGR2GRAY)
        img_new_data = img_new/255
        cv2.imshow('ewq',img_new_data)
        img_new_data = np.float32(np.resize(img_new,(1,32,32,3)))
        graph = tf.get_default_graph()
        x_new = graph.get_tensor_by_name('x_data:0')
        y_new = graph.get_tensor_by_name('y:0')
        pre = sess.run(y_new,feed_dict={x_new:img_new_data})
        y_position = np.argmax(pre)    
        y_p = np.max(pre)
        text = Name[y_position]+'%.4f'%(y_p*100)+'%'
        cv2.putText(frame,text,(40,50),cv2.FONT_HERSHEY_PLAIN,2.0, (0, 0, 255), 2)
    else:
        text = 'unknow'
        cv2.putText(frame,text,(40,50),cv2.FONT_HERSHEY_PLAIN,2.0, (0, 0, 255), 2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == 27 :
        break
cap.release()
cv2.destroyAllWindows() 
