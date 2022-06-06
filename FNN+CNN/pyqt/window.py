# from statistics import mode
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtGui import QPen
from PyQt5.QtGui import QFont
from PIL import ImageGrab, Image, ImageEnhance
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout

import torch
from fnn import Net


class MyMnistWindow(QWidget):

    def __init__(self):
        super(MyMnistWindow, self).__init__()

        self.model = torch.load('model_fnn.pth', map_location=torch.device('cpu'))

        self.setWindowTitle("手写识别")
        self.resize(600, 700)  # 设置窗口宽高
        self.setMouseTracking(False)

        self.pos_xy = []  #保存鼠标移动过的点

        # 添加一系列控件

        self.label_draw = QLabel('', self)      # 创建标签控件对象
        self.label_draw.setStyleSheet("QLabel{border:2px solid black;}")    # 设置样式
        self.label_draw.setAlignment(Qt.AlignCenter)    # 消除空隙

        self.label_result_name = QLabel('结果：', self)     # 设置标签内容
        self.label_result_name.setAlignment(Qt.AlignCenter)

        self.label_result = QLabel('  ', self)           # 填充空内容
        self.label_result.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_result.setStyleSheet("QLabel{border:2px solid black;}")
        self.label_result.setAlignment(Qt.AlignCenter)

        self.btn_recognize = QPushButton("识别", self)
        self.btn_recognize.clicked.connect(self.btn_recognize_on_clicked)   # 信号槽连接，下同

        self.btn_clear = QPushButton("清空", self)
        self.btn_clear.clicked.connect(self.btn_clear_on_clicked)

        self.btn_close = QPushButton("关闭", self)
        self.btn_close.clicked.connect(self.btn_close_on_clicked)

        self.top_layout = QHBoxLayout()
        self.bottom_layout = QHBoxLayout()
        self.main_layout = QVBoxLayout()
        self.top_layout.addWidget(self.label_draw)
        self.bottom_layout.addWidget(self.label_result_name)
        self.bottom_layout.addWidget(self.label_result)
        self.bottom_layout.addWidget(self.btn_recognize)
        self.bottom_layout.addWidget(self.btn_clear)
        self.bottom_layout.addWidget(self.btn_close)
        self.main_layout.addLayout(self.top_layout)
        self.main_layout.addLayout(self.bottom_layout)

        self.setLayout(self.main_layout)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen()
        pen.setColor(Qt.black)
        pen.setWidth(13)
        pen.setStyle(Qt.SolidLine)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)

        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                # 判断是否是断点
                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    def mouseMoveEvent(self, event):
        '''
            按住鼠标移动事件 将当前点添加到pos_xy列表中
        '''
        #中间变量pos_tmp提取当前点
        pos_tmp = (event.pos().x(), event.pos().y())
        #pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        '''
            鼠标按住后松开的事件
            在每次松开后向pos_xy列表中添加一个断点(-1, -1)
        '''
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        self.update()

    # 识别按钮的功能：截屏手写数字并将截图转换成28*28像素的图片，之后调用识别函数并显示识别结果
    def btn_recognize_on_clicked(self):
        self_pos = self.pos()
        bbox = (self_pos.x() + 40, self_pos.y() + 90,
                self_pos.x() + 580, self_pos.y() + 600)
        im = ImageGrab.grab(bbox)    # 截屏，手写数字部分
        im = im.resize((28, 28), Image.ANTIALIAS)  # 将截图转换成 28 * 28 像素
        # im.show()

        recognize_result = self.recognize_img(im)  # 调用识别函数

        self.label_result.setText(str(recognize_result))  # 显示识别结果
        self.update()

    # 清除按钮的功能：列表置空，识别结果一栏清空
    def btn_clear_on_clicked(self):
        self.pos_xy = []
        self.label_result.setText('')
        self.update()

    # 关闭按钮的功能：关闭窗口
    def btn_close_on_clicked(self):
        self.close()

    # 识别函数
    def recognize_img(self, img):  # 手写体识别函数
        myimage = img.convert('L')  # 转换成灰度图
        myimage = ImageEnhance.Contrast(myimage).enhance(15.0)
        myimage = ImageEnhance.Sharpness(myimage).enhance(0.5)
        # myimage.show()
        tv = list(myimage.getdata())  # 获取图片像素值
        tv = [(255 - x) for x in tv]
        # tva = [(255 - x) * 1.0 / 255.0 for x in tv]  # 转换像素范围到[0 1], 0是纯白 1是纯黑
        tv = torch.tensor(tv).float()
        # tv = tv.view(1, 28, 28)
        temp = self.model(tv)
        return int(torch.argmax(temp))