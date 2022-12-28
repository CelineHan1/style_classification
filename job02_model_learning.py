import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow


# help(model.compile)

X_train, X_test, Y_train, Y_test =np.load('./dataset/male_fashion_sample_data.npy', allow_pickle=True)
print(Y_train[0])
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

label = ['Bohemian','casual','military','modern','punk','retro']

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(220, 220, 3), padding='same', activation='relu'))
model.add(MaxPool2D(padding='same', pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(padding='same', pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_accuracy', patience=7)
fit_his=model.fit(X_train, Y_train, batch_size=64, epochs=10, validation_split=0.15,callbacks=[early_stopping])
score=model.evaluate(X_test, Y_test)
print('Evaluation loss:', score[0])
print('Evaluation accuracy:', score[1])
model.save('./models/fashion_{}.h5'.format(str(np.around(score[1],2))[-2:]))
plt.plot(fit_his.history['accuracy'],label='accuracy')
plt.plot(fit_his.history['val_accuracy'],label='val_accuracy')
plt.show()
plt.plot(fit_his.history['loss'],label='loss')
plt.plot(fit_his.history['val_loss'],label='val_loss')
plt.legend()
plt.show()