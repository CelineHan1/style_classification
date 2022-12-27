from PIL import Image     #설치시에는 pip pillow 사용
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test =np.load('./datasets/binary_fashion_data.npy', allow_pickle=True)
print(Y_test)
model= load_model('./fashion_85.h5')
label = ['military','punk','retro','street']
my_sample = np.random.randint(100)
print(my_sample)
plt.imshow(X_test[my_sample], cmap='gray')
plt.show()
print(label[np.argmax(Y_test[my_sample])])
pred = model.predict(X_test[my_sample].reshape(1, 64, 64, 3))
print(pred)
print(label[np.argmax(pred)])