'''
构建神经网络
'''
from tensorflow.python.keras import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D

class cnn_mnsit:#虽然是糟糕的名字，但无所谓了
    def __init__(self,class_num = 10,img_size = 28) -> None:
        self.class_num = class_num#数据类型数量
        self.img_size = img_size#图片大小，方的
    def create_model(self):#
        self.model = Sequential()#顺序链接

        #开卷，32个卷积层 3*3卷积核 加上边界使得图片大小不变     relu作为激活函数       输入的形状
        self.model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(self.img_size,self.img_size,1)))#单通道
        self.model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))#数字基本没有纹理特征，统统选用maxPooling
        self.model.add(Dropout(0.2))#随机丢弃部分数据，避免过拟合

        #再来两层
        self.model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
        self.model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
        self.model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.5))

        #扁平化
        self.model.add(Flatten())
        self.model.add(Dense(516,activation='relu'))#全连接层
        self.model.add(Dropout(0.6))
        self.model.add(Dense(self.class_num,activation='softmax'))#输出层

    def compile(self):
        #损失函数使用categorical_crossentropy，梯度下降算法使用adam,正确率作为评价标准
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
    