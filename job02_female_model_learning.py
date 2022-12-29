import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow


# help(model.compile)
size = 64
epochs=15

X_train =np.load('./dataset/female_style_img_size_{}x{}_X_train.npy'.format(size,size), allow_pickle=True)
X_test =np.load('./dataset/female_style_img_size_{}x{}_X_test.npy'.format(size,size), allow_pickle=True)
Y_train =np.load('./dataset/female_style_img_size_{}x{}_Y_train.npy'.format(size,size), allow_pickle=True)
Y_test =np.load('./dataset/female_style_img_size_{}x{}_Y_test.npy'.format(size,size), allow_pickle=True)
print(Y_train[0])
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

label=['bohemian','casual','military','modern','punk','retro']


model = Sequential()
model.add(Conv2D(32, kernel_size=(8, 8), input_shape=(size, size, 3), padding='same', activation='relu'))
model.add(MaxPool2D(padding='same', pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPool2D(padding='same', pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_accuracy', patience=7)
fit_his=model.fit(X_train, Y_train, batch_size=32, epochs=epochs, validation_split=0.1,callbacks=[early_stopping])
score=model.evaluate(X_test, Y_test)
print('Evaluation loss:', score[0])
print('Evaluation accuracy:', score[1])
# model.save('./models/female_fashion_{}.h5'.format(str(np.around(score[1],2))[-2:]))
# model.save('./models/female_fashion_size_{}_acc_{}.h5'.format(size,str(np.around(score[1],2))[-2:]))
print('save : ','female_fashion_size_{}_acc_{}.h5'.format(size,str(np.around(score[1],2))[-2:]))
plt.plot(fit_his.history['accuracy'],label='accuracy')
plt.plot(fit_his.history['accuracy'],label='val_accuracy')
plt.legend()
plt.show()
plt.plot(fit_his.history['loss'],label='loss')
plt.plot(fit_his.history['val_loss'],label='val_loss')
plt.legend()
plt.show()