from PIL import Image
from array import *
import numpy as np

class convertImgToMNIST(object):
    def __init__(self,url = './temp.png') -> None:
        self.url = url
        self.width = 28
        self.height = 28
    def getArray(self):
        img = Image.open(self.url)
        img_list = []
        arr_list = []
        #print(img)
        pixel = img.load()
        for x in range(0,self.width):
            for y in range(0,self.height):
                img_list.append(self.pix_mean(pixel[y,x]))
            arr_list.append(np.array(img_list,dtype=np.uint8))
            img_list.clear()
        return np.array(arr_list)
    def pix_mean(self,pix):
        try:
            return self.mean(pix[0],pix[1],pix[2])
        except:
            return pix#此时本身就是单通道
    def mean(self,r,g,b):#转为单通道
        return 0.5*(255-r)+0.3*(255-g)+0.2*(255-b)#mnist黑色是最亮的(一般的笔都是黑色),这儿做个取反处理