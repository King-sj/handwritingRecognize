from keras.datasets import mnist
from sklearn.naive_bayes import GaussianNB
import joblib
#加载数据集
(train_x,train_y),(test_x,test_y) = mnist.load_data()
train_x = train_x.reshape(train_x.shape[0],28*28)
test_x = test_x.reshape(test_x.shape[0],28*28)
model = GaussianNB()
model.fit(train_x,train_y)
joblib.dump(model,"GaussianNB.m")
print(model.score(test_x,test_y))
