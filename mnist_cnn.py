#coding:utf-8
#导入所用模块
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

#初始化模型常数
INPUT_SIZE = 784
IMG_HEIGHT = 28
IMG_WIDTH = 28
LEARN_RATE = 0.0001
TRAINING_STEP = 100000
BATCH_SIZE = 50

def save2img(mnist):
    import os
    import scipy.misc
    save_path = './MNIST_data/raw/'
    #创建图片文件夹
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    #保存图片
    for i in range(20):
        image_array = mnist.train.images[i,:]
        image_array = image_array.reshape(28,28)
        filename = save_path + 'mnist_train_%d.jpg' % i
        print(str(i) + '-' + str(np.argmax(mnist.train.labels[i])))
        scipy.misc.toimage(image_array,cmin=0.0,cmax=1.0).save(filename)

#变量初始化函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#巻积函数
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
#池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def mnist_train(mnist):
    #构建模型
    x = tf.placeholder(tf.float32,[None,INPUT_SIZE],name='input-x')
    y_ = tf.placeholder(tf.float32,[None,10])
    x_img = tf.reshape(x,[-1,IMG_HEIGHT,IMG_WIDTH,1]) #tf.nn.conv2d input: [batch,in_height,in_width,in_channels]

    #第一层巻积
    w_conv1 = weight_variable([5,5,1,32])  #tf.nn.conv2d filter: [filter_height, filter_width, in_channels, out_channels]
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_img,w_conv1) + b_conv1) # 计算巻积
    h_pool1 = max_pool_2x2(h_conv1) #池化层

    #第二层巻积
    w_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #全链接层(输出1024维)
    w_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

    #使用dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    #全链接层将1024维转化为10维，对应十个类别
    w_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop,w_fc2)+b_fc2

    #定义损失函数（交叉熵）
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))

    #训练步
    train_step = tf.train.AdamOptimizer(LEARN_RATE).minimize(cross_entropy)

    #定义测试准确率
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    #创建会话并迭代训练
    with tf.Session() as sess:
        #初始化变量
        tf.global_variables_initializer().run()
        #迭代训练
        for i in range(TRAINING_STEP):
            if i%100 ==0:
                accuracy_temp = np.zeros(200)
                for j in range(200):
                    accuracy_temp[j] = sess.run(accuracy,feed_dict={x:mnist.test.images[j*50:(j+1)*50].reshape([-1,784]),y_:mnist.test.labels[j*50:(j+1)*50].reshape([-1,10]),keep_prob:1})
                print(accuracy_temp.mean())
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step,feed_dict={x:xs,y_:ys, keep_prob:0.5})     
        

if __name__ == '__main__':
    mnist = input_data.read_data_sets('./MNIST_data',one_hot=True)
    #打印图片
    #save2img(mnist)
    #训练神经网络
    mnist_train(mnist)
