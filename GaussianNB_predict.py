import joblib
from convertImgToMNIST import *
class GaussianNB_predict:
    def __init__(self) -> None:
        self.model = joblib.load("GaussianNB.m")
    def loadData(self,url='./temp.png'):
        self.test_x = convertImgToMNIST(url).getArray()
        self.test_x = self.test_x.reshape(1,28*28)
    def predict(self):
        # try:
        #     self.load_data()
        # except:
        #     print("couldn't find test data")
        self.loadData()
        res = self.model.predict(self.test_x)
        return res