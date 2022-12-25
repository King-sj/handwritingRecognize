import pickle
import numpy as np
from itertools import chain
from knn_convertImgToMNIST import *
from PIL import Image

class knn_predict:

    def __init__(self) -> None:
        self.test_x = []
        with open('knn_model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def load_data(self, url='./temp.png'):
        self.test_x = knn_convertImgToMNIST(url).getArray()

    def predict(self):
        try:
            self.load_data()
        except:
            print("couldn't find test data")
        res = self.model.predict([self.test_x])
        print(res[0])
        return res[0]

# mod = knn_predict()
# mod.load_data()
# ans, acc = mod.predict()
# print(ans, acc)