
from Ui_mainWindow import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
from convertImgToMNIST import *
from cnn_predict import *
from knn_predict import *
from svm_predict import *
from GaussianNB_predict import *
from Decisiontree_predict import *
class HandWritingRecognitionWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent = None):
        super(HandWritingRecognitionWindow,self).__init__(parent)
        ###########################窗口设置#################################
        self.setupUi(self)
        self.setWindowTitle("手写数字识别")
        self.setFixedSize(self.width(),self.height())
        #############################创建数据成员##################################
        self.pixmap = QPixmap(self.width(),self.height())
        self.pixmap.fill(Qt.white)

        self.painter = QPainter()

        self.beginPoint = QPoint(0,0)
        self.endPoint = QPoint(0,0)
        self.flag = 0
        self.pointList = []#记录

        self.cnn_model = cnn_predict()#创建训练好的模型
        self.knn_model = knn_predict()
        self.svm_model = svm_predict()
        self.GaussianNB_model = GaussianNB_predict()
        self.Decisiontree_model = Decisiontree_predict()
        ###############################################################################

        ###########################事件绑定########################################
        self.actionrecognize.triggered.connect(self.dealRecognize)
        self.actionimport.triggered.connect(self.dealImportImg)
        self.actionrewrite.triggered.connect(self.dealRewrite)
        self.actionundo.triggered.connect(self.dealUndo)

    def init_Painter(self):
        self.Pen = QPen(Qt.SolidLine)
        self.Pen.setColor(QColor(0,0,255))#red
        self.Pen.setWidth(30)#mnist数据大小是28*28，笔太细会导致缩小后差距太大
        self.painter.setPen(self.Pen)
        self.painter.setBrush(Qt.white)
    ###############控件事件处理##################################
    def dealRecognize(self):#识别
        self.toImg()

        #self.cnn_model.load_data()#加载测试数据
        cnn_ans = self.cnn_model.predict()
        knn_ans = self.knn_model.predict()
        svm_ans = self.svm_model.predict()
        Gau_ans = self.GaussianNB_model.predict()
        Dec_ans = self.Decisiontree_model.predict()
        self.result = 'the number is: cnn:{},knn:{},svm:{},GuassianNB{},Decisiontree{}'.format(cnn_ans,
        knn_ans,svm_ans,Gau_ans,Dec_ans)
        self.labelResult.setText(self.result)


    def dealUndo(self):#撤销
        try:
            self.pixmap.fill(Qt.white)#清空
            self.pointList.pop()#删除最后一步
            self.painter.begin(self.pixmap)
            self.init_Painter()
            for list in self.pointList:
                s = list[0]
                for t in list[1:-1]:
                    self.painter.drawLine(s,t)
                    s = t
                    self.update()
            self.painter.end()
            self.update()
        except:
            if self.pointList.count == 0:
                print("the img is empty")
                self.pixmap.fill(Qt.white)
            else:
                print(self.pointList)
                print("undo error")

    def dealRewrite(self):#重写
        self.pixmap.fill(Qt.white)
        self.pointList.clear()
    def dealImportImg(self):#导入图片
        self.img_url,self.img_type = QFileDialog.getOpenFileName(self,'打开文件','.','图像文件(*.jpg *.png)')

        pix = QtGui.QPixmap(self.img_url)
        self.pixmap = pix.scaled(self.label_writingBoard.width(),self.label_writingBoard.height())
        self.update()
        self.pixmap = pix        
        self.dealRecognize()
        #print(self.img_url)

    def toImg(self):
        path = './temp.png'
        try:
            os.makedirs(os.path.dirname(path))
        except:
            print("makedir error or file exist")
        pix = self.pixmap.scaled(28,28)
        pix.save(path)
    #######################系统事件处理########################
    def paintEvent(self, e: QtGui.QPaintEvent) -> None:
        self.label_writingBoard.setPixmap(self.pixmap)
        return super().paintEvent(e)
    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == Qt.LeftButton :
            self.flag = True
            self.beginPoint = e.pos() - self.label_writingBoard.pos() - QPoint(1,16)#偏移一点，看起来自然

            self.pointList.append([self.beginPoint])

            self.endPoint = e.pos()- self.label_writingBoard.pos()-QPoint(2,16)
        return super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        if Qt.LeftButton and self.flag:            
            self.painter.begin(self.pixmap)
            self.init_Painter()    
            self.endPoint = e.pos()- self.label_writingBoard.pos()-QPoint(2,16)

            self.pointList[-1].append(self.endPoint)

            self.painter.drawLine(self.beginPoint,self.endPoint)
            self.beginPoint = self.endPoint
            self.painter.end()
            self.update()
        return super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        self.flag = False
        self.beginPoint = QPoint(0,0)
        self.endPoint = QPoint(0,0)

        return super().mouseReleaseEvent(e)
    ##########################################################################