import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PIL import Image
from keras.models import load_model
import numpy as np
import random
from random import sample
import glob


form_window = uic.loadUiType('./cat_and_dog.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.model_male = load_model('models/male_fashion_size_64_acc_69_ep_15.h5')
        self.model_female = load_model('models/female_fashion_size_64_acc_42.h5')
        self.path=('./style_img/', '')
        self.label=['bohemian','casual','military','modern','punk','retro']




        self.btn_open.clicked.connect(self.image_open_slot)
        self.btn_recommend.clicked.connect(self.btn_recommend_slot)

    def image_open_slot(self):
        self.path_01 = QFileDialog.getOpenFileName(self, 'Open File', './style_img/test_img/', 'Image Files(*.jpg;*.png);;All Files(*.*)')
        print(self.path_01)
        if self.path_01[0]:
            pixmap = QPixmap(self.path_01[0])
            self.lbl_image.setPixmap(pixmap)
            # 남자 모델 실행
            if self.chbox_male.isChecked() == True:
                try:
                    img=Image.open(self.path_01[0])
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
                    img=Image.open(self.path_01[0])
                    print('debug01')
                    img=img.convert('RGB')
                    img=img.resize((64,64))
                    data=np.asarray(img)
                    data=data/255
                    data=data.reshape(1,64,64,3)
                    pred=self.model_female.predict(data)
                    print('')
                    self.lbl_pred.setText(self.label[np.argmax(pred)])
                    print(self.label[np.argmax(pred)])
                except:
                    print('error')

    def btn_recommend_slot(self):

        print('path', self.path)
        print('path', self.path_01)
        self.sex = self.check_sex()
        self.style = self.lbl_pred.text()
        self.dir = self.path[0] + self.check_sex() + '/'+ self.style +'/'
        print('debug01',self.dir)

        self.files = glob.glob(self.dir + '*.jpg')
        print('debug02', len(self.files))
        self.ranint5 = sample(range(len(self.files)), 3)
        print('debug03', self.ranint5)
        for i in range(3):
            print('i :', i)
            random_img = self.files[self.ranint5[i]]
            # random_img = '{}\{}'.format(self.dir,self.files[self.ranint5[i]])
            print('debug04', random_img)
            # img = Image.open(random_img)
            # print('debug05')
            if i == 0:
                print('if 0 start')
                pixmap = QPixmap(random_img)
                self.lbl_showimg01.setPixmap(pixmap)
                print('if 0 end')

            if i == 1:
                print('if 1 start')
                pixmap = QPixmap(random_img)
                self.lbl_showimg02.setPixmap(pixmap)
                print('if 1 end')

            else:
                print('if 2 start')
                pixmap = QPixmap(random_img)
                self.lbl_showimg03.setPixmap(pixmap)
                print('if 2 end')


    def check_sex(self):
        if self.chbox_male.isChecked() == True:
            return 'male'
        else:
            return 'female'


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())


def uiUpdate():
    # UI 파일이 있는 경로로 지정
    path = glob.glob('./*.ui')
    for ui_path in path:
        ui_ = open(ui_path, 'r', encoding='utf-8')
        lines_ = ui_.readlines()
        ui_.close()
        for ii, i in enumerate(lines_):
            if 'include location' in i:
                lines_[ii] = i.replace('.qrc', '.py')

        ui_ = open(ui_path, 'w', encoding='utf-8')
        [ui_.write(i) for i in lines_]
        ui_.close()
        print('{} update'.format(ui_path))




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