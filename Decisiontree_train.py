from sklearn import tree
from keras.datasets import mnist
import joblib
#加载数据集
(train_x,train_y),(test_x,test_y) = mnist.load_data()
train_x = train_x.reshape(train_x.shape[0],28*28)
test_x = test_x.reshape(test_x.shape[0],28*28)
#创建并训练
model = tree.DecisionTreeClassifier()
model.fit(train_x,train_y)
#保存模型
joblib.dump(model,"Decisiontree.m")
print(model.score(test_x,test_y))