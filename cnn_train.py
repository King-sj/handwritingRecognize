from keras.datasets import mnist#导入手写数字的数据集
from cnn_util import *
from keras.utils import np_utils
#加载数据集
(train_x,train_y),(test_x,test_y) = mnist.load_data()

# print(train_x[0])
# print(train_y[0])
# print(test_x[0])
# print(test_y[0])
#词穷，想不到名字
ob = cnn_mnsit()#实例化
ob.create_model()#创建模型
ob.compile()#编译模型

#处理数据，转变为合适的形状
#                         降一维维数组数量 28*28单通道
train_x = train_x.reshape(train_x.shape[0],28,28,1)
test_x = test_x.reshape(test_x.shape[0],28,28,1)

#数据预处理为 float32 映射到 0 -> 1
# train_x = train_x.astype('float32')
# test_x = test_x.astype('float32')
# train_x /= 255
# test_x /= 255
# 将1维类数组转换为10维类矩阵
# shape 从 (60000,) -> (60000,10)
train_y = np_utils.to_categorical(train_y,10)
test_y = np_utils.to_categorical(test_y,10)

#开始训练
#32个为一组，按批更新权重  训练10次先    将数据打乱
ob.model.fit(train_x,train_y,batch_size=32,epochs=10,shuffle=True)
ob.model.save('./sj_mnist.h5')#保存好模型参数
#评估
score = ob.model.evaluate(test_x,test_y)