import tensorflow as tf
import numpy as np

tf.reset_default_graph()
data_tr = np.load('./data_train.npy')
data_te = np.load('./data_test.npy')
labels_tr = np.load('./labels_train.npy')
labels_te = np.load('./labels_test.npy')

#数据集处理
#将标签以独热编码的形式读入数据

data_tr,data_te= np.float32(data_tr),np.float32(data_te)
labels_train,labels_test = tf.one_hot(labels_tr,10),tf.one_hot(labels_te,10)

#img_data=data_tr[1,:,:,:]
img_data = tf.placeholder('float32',[None,32,32,3],name='x_data')
#img_new = tf.reshape(img_data,[1,32,32,3])
labels_data = tf.placeholder('float32',[None,10])
##卷积
#设置卷积操作filter
w1 = tf.Variable(tf.random_normal([5,5,3,32],stddev=0.01,dtype='float32'),name='w1')
w2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01,dtype='float32'),name='w2')
conv1 = tf.nn.conv2d(img_data,w1,strides=[1,1,1,1],padding='SAME')
conv1 = tf.nn.relu(conv1)
#池化
pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#再卷积
conv2 = tf.nn.conv2d(pool1,w2,strides=[1,1,1,1],padding='SAME')
conv2 = tf.nn.relu(conv2)
#再池化
pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
pool_tr = tf.reshape(pool2,[-1,8*8*64])

#权值 阈值设置
w3 = tf.Variable(tf.random_normal([8*8*64,32],stddev=0.01),dtype='float32')
bias1 = tf.Variable(tf.zeros([32]))
w4 = tf.Variable(tf.random_normal([32,10],stddev=0.01),dtype='float32')
bias2 = tf.Variable(tf.zeros([10]))
#w1 = tf.Variable(tf.random_normal([3,3,3,12],stddev=0.01,dtype='float32'),name='w1')     
#w2 = tf.Variable(tf.random_normal([2,2,12,24],stddev=0.01,dtype='float32'),name='w2')
#卷积函数conv2d(picture，filter,strides,padding)
#conv1 = tf.nn.conv2d(img_data,w1,strides=[1,1,1,1],padding='SAME')
#conv1 = tf.nn.relu(conv1)
##池化
#pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
##再卷积
#conv2 = tf.nn.conv2d(pool1,w2,strides=[1,1,1,1],padding='SAME')
#conv2 = tf.nn.relu(conv2)
##再池化
#pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#pool_tr = tf.reshape(pool2,[-1,8*8*24])
#
##权值 阈值设置
#w3 = tf.Variable(tf.random_normal([8*8*24,800],stddev=0.01),dtype='float32')
#bias1 = tf.Variable(tf.zeros([800]))
#w4 = tf.Variable(tf.random_normal([800,10],stddev=0.01),dtype='float32')
#bias2 = tf.Variable(tf.zeros([10]))
#输入层到隐层
H = tf.sigmoid(tf.matmul(pool_tr,w3)+bias1)
H = tf.nn.relu (H)
#隐层到输出层
y = tf.nn.softmax(tf.matmul(H,w4)+bias2,name='y')

#动态学习率
global_step = tf.Variable(0, trainable=False) # 可设动态学习率
learning_rate = 0.015#tf.train.exponential_decay(1e-4)

#交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels_data*tf.log(y),axis=1))


optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(cross_entropy,global_step=global_step)

init = tf.global_variables_initializer()

d_tr = np.zeros([500,32,32,3],dtype='float32')
l_tr = np.zeros([500,10],dtype='float32')


#构建计算图
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    labels_train,labels_test = sess.run([labels_train,labels_test])  
    for i in range(500):
        k=0
#        if i%100==0:
        pre = tf.equal(tf.argmax(y,axis=1),tf.argmax(labels_data,axis=1))
        acc,cost = sess.run([pre,cross_entropy],feed_dict = {img_data: data_te,labels_data:labels_test})
        print(i,'test acc:%.4f'%(sum(acc)/len(acc)),' cost:%.4f'%cost)
        for j in np.random.randint(0,4799,size=[500]):
            d_tr[k] = data_tr[j,:]
            l_tr[k] = labels_train[j]
            k = k+1
        acc,cost,_=sess.run([pre,cross_entropy,train],feed_dict={img_data:d_tr,labels_data:l_tr})
        print(i,'train acc:%.4f'%(sum(acc)/len(acc)),' cost:%.4f'%cost,'\n')
    saver.save(sess,'E:/spyder/cnn/model_1/cnnmodel')

    
    





































