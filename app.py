import sys
from Method_mainWindow import HandWritingRecognitionWindow
from PyQt5.QtWidgets import QApplication
if __name__ == "__main__":
    app = QApplication(sys.argv)
    Win = HandWritingRecognitionWindow()
    Win.show()
    sys.exit( app.exec_() )#进入事件循环