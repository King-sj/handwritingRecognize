import joblib
from convertImgToMNIST import *
class svm_predict:
    def __init__(self) -> None:
        self.model = joblib.load("svm.m")
    def loadData(self,url = './temp.png'):
        self.test_x = convertImgToMNIST(url=url).getArray()
        self.test_x = self.test_x.reshape(1,28*28)
    def predict(self):
        try:
            self.loadData()
        except:
            print("couldn't find test data")
        res = self.model.predict(self.test_x)
        return res