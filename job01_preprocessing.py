from PIL import Image     #설치시에는 pip pillow 사용
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

img_dir = './style_img/'
categories=['bohemian male','casual male','military male','modern male','punk male','retro male','bohemian female','casual female','military female','modern female','punk female','retro female']
image_w=110
image_h=110
pixel=image_h*image_w*3
X=[]
Y=[]
files=None
print(list(enumerate(categories)))
for idx, category in enumerate(categories):
    files=glob.glob(img_dir+category+'/*.jpg')
    print(category,len(files))
    print(files[:3])
    for i, f in enumerate(files):
        try:
            img=Image.open(f)
            # print(img)
            # img.show()
            img=img.convert('RGB')
            img=img.resize((image_w,image_h))
            data=np.asarray(img)
            X.append(data)
            Y.append(idx)
            if i % 300 == 0:
                print(category, ':', f)
        except:
             print('error',f)
X=np.array(X)
Y=np.array(Y)

y = np.array(Y).reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
encoded_y = encoder.fit_transform(y)
print(encoded_y)

X= X/255
print(X[0])
print(Y[0])
X_train, X_test, Y_train, Y_test = train_test_split(X,encoded_y,test_size=0.2)
# xy=(X_train, X_test, Y_train, Y_test)
np.save('./dataset/style_img_size_{}x{}_X_train.npy'.format(image_w,image_h),X_train)
np.save('./dataset/style_img_size_{}x{}_X_test.npy'.format(image_w,image_h),X_test)
np.save('./dataset/style_img_size_{}x{}_Y_train.npy'.format(image_w,image_h),Y_train)
np.save('./dataset/style_img_size_{}x{}_Y_test.npy'.format(image_w,image_h),Y_test)