import tensorflow as tf 
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('观赏型花卉的智能分类识别')
        self.model = tf.keras.models.load_model("models/restnet_flower.h5")
        self.to_predict_name = "images/init.png"
        self.class_names = ['雏菊', '蒲公英','玫瑰','向日葵', '郁金香']
        self.resize(400, 400)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('宋体', 10)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("花卉的智能分类识别")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.png', img_init)
        self.img_label.setPixmap(QPixmap('images/target.png'))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 选择一张图片 ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" 开始识别")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)

        self.result = QLabel("待识别....")
        self.result.setFont(QFont('宋体', 10))
        right_layout.addStretch()
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        # right_layout.addSpacing(5)
        right_widget.setLayout(right_layout)


        label_super = QLabel()
        label_super.setFont(QFont('楷体', 12))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setObjectName("color")
        main_widget.setStyleSheet("#color{background-color:pink}")
        main_widget.setLayout(main_layout)

        self.addTab(main_widget, '主页面')

    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', 'Image files(*.jpg , *.png)')
        img_name = openfile_name[0]
        if img_name == '':
            pass
        else:
            try:
                self.to_predict_name = img_name
                img_init = cv2.imread(self.to_predict_name)
                img_init = cv2.resize(img_init, (224, 224))
                cv2.imwrite('images/target.png', img_init)
                self.img_label.setPixmap(QPixmap('images/target.png'))
            except (FileNotFoundError, OSError) as e:
                self.to_predict_name = None
                QMessageBox.warning(self, "错误", "文件读取错误：" + str(e))

    def predict_img(self):
        if self.to_predict_name is None:
            QMessageBox.warning(self, "错误", "请先选择图片！")
            return
        try:
            img = Image.open('images/target.png')
            img = np.asarray(img)
            # gray_img = img.convert('L')
            # img_torch = self.transform(gray_img)
            outputs = self.model.predict(img.reshape(1, 224, 224, 3))
            # print(outputs)
            result_index = np.argmax(outputs)
            result = self.class_names[result_index]
            self.result.setText(result)
        except (FileNotFoundError, OSError) as e:
            QMessageBox.warning(self, "错误", "文件读取错误：" + str(e))


        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())



