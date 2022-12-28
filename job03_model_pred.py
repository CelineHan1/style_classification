from PIL import Image     #설치시에는 pip pillow 사용
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test =np.load('./dataset/male_fashion_sample_data.npy', allow_pickle=True)
print(Y_test)
model= load_model('./fashion_66.h5')
label = ['Bohemian','casual','military','modern','punk','retro']
my_sample = np.random.randint(100)
print(my_sample)
plt.imshow(X_test[my_sample], cmap='gray')
plt.show()
print(label[np.argmax(Y_test[my_sample])])
pred = model.predict(X_test[my_sample].reshape(1, 220, 220, 3))
print(pred)
print(label[np.argmax(pred)])