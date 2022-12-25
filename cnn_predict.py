from cnn_util import *
from convertImgToMNIST import *

class cnn_predict:
    def __init__(self,weights_url = './cnn_mnist.h5') -> None:
        self.mnist = cnn_mnsit()
        self.mnist.create_model()
        self.mnist.model.load_weights(weights_url)
    def load_data(self,url = './temp.png'):
        self.test_x = convertImgToMNIST(url=url).getArray()
        #                     就一个预测数据
        self.test_x = self.test_x.reshape(1,28,28,1)
    def predict(self):
        try:
            self.load_data()
        except:
            print("couldn't find test data")
        res = self.mnist.model.predict(self.test_x)
        res = res[0]
        m = -1
        i = 0
        ans = -1
        #sum = 0
        #print(res)
        for acc in res:
            #sum += acc
            #print("acc {}".format(acc))
            if acc > m:
                m = acc
                #print("max p :{}".format(m))
                ans = i
            i+=1
        # #print(ans)
        # accuracy = m/sum
        return ans#答案