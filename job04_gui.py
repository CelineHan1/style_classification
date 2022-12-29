import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PIL import Image
from keras.models import load_model
import numpy as np

form_window = uic.loadUiType('./cat_and_dog.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.model_male = load_model('models/male_fashion_size_64_acc_69_ep_15.h5')
        self.model_female = load_model('models/female_fashion_size_64_acc_42.h5')
        self.path=('../datasets/cat_dog/train/cat.4.jpg', '')
        self.label=['bohemian','casual','military','modern','punk','retro']

        self.btn_open.clicked.connect(self.image_open_slot)

    def image_open_slot(self):
        self.path = QFileDialog.getOpenFileName(self, 'Open File', './style_img/test_img/', 'Image Files(*.jpg;*.png);;All Files(*.*)')
        print(self.path)
        if self.path[0]:
            pixmap = QPixmap(self.path[0])
            self.lbl_image.setPixmap(pixmap)
            if self.chbox_male.isChecked() == True:
                try:
                    img=Image.open(self.path[0])
                    print('debug01')
                    img=img.convert('RGB')
                    img=img.resize((64,64))
                    data=np.asarray(img)
                    data=data/255
                    data=data.reshape(1,64,64,3)
                    pred=self.model_male.predict(data)
                    print('debug02')
                    self.lbl_pred.setText(self.label[np.argmax(pred)])
                    print(self.label[np.argmax(pred)])
                except:
                    print('error')
            else:
                try:
                    img=Image.open(self.path[0])
                    print('debug01')
                    img=img.convert('RGB')
                    img=img.resize((64,64))
                    data=np.asarray(img)
                    data=data/255
                    data=data.reshape(1,64,64,3)
                    pred=self.model_female.predict(data)
                    print('debug02')
                    self.lbl_pred.setText(self.label[np.argmax(pred)])
                    print(self.label[np.argmax(pred)])
                except:
                    print('error')
    #
    # def run_model(self):




if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())




# import sys          # sys : 파이썬 기본 라이브러리,
# from PyQt5.QtWidgets import *           # pyQt5 라이브러리 안의 모든 것을 import 하는 법
# from PyQt5 import uic
#
# form_window = uic.loadUiType('./fassion_classification.ui')[0]      #ui를 class로 만들어줌   # 파일은 designer에서 만들어 프로젝트파일 안에 넣는다.
#
# class Exam(QWidget, form_window):           # 다중상속받음
#     def __init__(self):                     #
#         super().__init__()
#         self.setupUi(self)                  # ui 초기화
#
# if __name__ == "__main__":                  # 모듈로 사용할 수 도 있으니 습관적으로 만들어주어라.
#     app = QApplication(sys.argv)
#     mainWindow = Exam()                     # 클래스의 생성자를 호출하여 mainWindow 변수에 저장
#     mainWindow.show()                       # mainWindow를 보여줌
#     sys.exit(app.exec_())