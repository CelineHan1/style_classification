from PIL import Image     #설치시에는 pip pillow 사용
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt

# X_train, X_test, Y_train, Y_test =np.load('./dataset/style_img_size_110x110.npy', allow_pickle=True)
X_train =np.load('./dataset/male_style_img_size_64x64_X_train.npy', allow_pickle=True)
X_test =np.load('./dataset/male_style_img_size_64x64_X_test.npy', allow_pickle=True)
Y_train =np.load('./dataset/male_style_img_size_64x64_Y_train.npy', allow_pickle=True)
Y_test =np.load('./dataset/male_style_img_size_64x64_Y_test.npy', allow_pickle=True)
print(Y_test)
model= load_model('./models/male_fashion_size_64_acc_69_ep_15.h5')
# label=['bohemian male','casual male','military male','modern male','punk male','retro male','bohemian female','casual female','military female','modern female','punk female','retro female']
label=['bohemian','casual','military','modern','punk','retro']
my_sample = np.random.randint(150)
print(my_sample)
print(label[np.argmax(Y_test[my_sample])])
pred = model.predict(X_test[my_sample].reshape(1, 64, 64, 3))
print(label)
print(pred)
print(label[np.argmax(pred)])
plt.imshow(X_test[my_sample], cmap='gray')
plt.show()