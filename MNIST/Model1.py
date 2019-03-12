'''
LeNet-1
'''

# usage: python MNISTModel1.py - train the model

from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras import backend as K

import numpy as np

def load_data(path="MNIST_data/mnist.npz"): # 得额外加，文件里没有mnist集合
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def Model1(input_tensor=None, train=False):
    print("-------------------Testing on LeNet-1(52 neurons)---------------------")
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)
    
    # 重新从 minist中导入数据
    if train: # False： 不需要
        batch_size = 256
        nb_epoch = 10

        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, shuffled and split between train and test sets
        # (x_train, y_train), (x_test, y_test) = mnist.load_data()
        (x_train, y_train), (x_test, y_test) = load_data() # 在上面 预定义好了 MNIST_data/mnist.npz

        print(x_train.shape)
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)
        
        
        # 实例化 Keras 张量
        # shape: 一个!!尺寸元组（整数），不体现批量大小!!
        input_tensor = Input(shape=input_shape)
    elif input_tensor is None:
        print('you have to proved input_tensor when testing')
        exit()
    
    print("in LeNet-1 input_tensor = ",input_tensor)
    
    
    # step1： define the model and layers, 
    # two main types of models： 1 Sequential model 2 Model class used with the functional API.
    # 这里用Model class API的写法： 有input tensor(s) and output tensor(s) 即可， 需要的layers 在output中定义
    
    # block1
    # print("in Model1 input_tensor = ",input_tensor)
    x = Convolution2D(4, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    # print("in Model1 x = ", x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Convolution2D(12, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x) # 输入展平。不影响批量大小---但没有设置下一层，因为
    # 分开写，方便修改激活函数(dense的activation属性不指定，则不使用激活函数 (即，「线性」激活: a(x) = x))
    x = Dense(nb_classes, name='before_softmax')(x) # last layer: 全连接层---flatten层 下一层的output size 即neuron数量:  (*, nb_classes)
    x = Activation('softmax', name='predictions')(x) # 将激活函数应用于输出

    # type: keras.models. 是Sequential()： 因为直接调用API， input一个张量，output 也是张量的（但需要我们定义嵌套型的layer结构）
    # 是未训练的model！！！ (weight是初始的， 后面train或者从Model1.h5获取!!!!）
    model = Model(input_tensor, x)  

    if train: 
        # step2: compiling, 对学习过程 进行配置
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        # step3： trainig
        model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)
        # save model: the weights, 与 evaluation的结果 无关
        model.save_weights('./Model1.h5')
        
        # step4: evaluate（just a score)
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
    else: # use the pre-trained weights
        model.load_weights('./Model1.h5')
        print('Model1:LeNet-1  loaded')

    # K.clear_session()

    return model 


if __name__ == '__main__':
    Model1(train=True)
