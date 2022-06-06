import sys
from PyQt5.QtWidgets import QApplication
from window import MyMnistWindow
from fnn import Net

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mymnist = MyMnistWindow()   # 实例化主窗体
    mymnist.show()              # 展示主船体
    app.exec_()