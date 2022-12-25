from keras.datasets import mnist#导入手写数字的数据集
from sklearn import svm
import joblib
#加载数据集
(train_x,train_y),(test_x,test_y) = mnist.load_data()

train_x = train_x.reshape(train_x.shape[0],28*28)
test_x = test_x.reshape(test_x.shape[0],28*28)
#创建模型
svm_model = svm.SVC()
#训练模型
svm_model.fit(train_x,train_y)

#保存
joblib.dump(svm_model,"svm.m")
#测试
print(svm_model.score(test_x,test_y))