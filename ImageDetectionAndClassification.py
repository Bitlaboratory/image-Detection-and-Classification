

from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,Input 
from keras.models import Sequential,Model,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers  import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import K
import matplotlib.pyplot as plt
import numpy as np
ClassNum=3
Trainpath='D:/AI/Keras/DataSet/BeesVsInsects/insects/Train'
TestPath='D:/AI/Keras/DataSet/BeesVsInsects/insects/Test'
PredictPath='D:/AI/Keras/DataSet/BeesVsInsects/insects/Predict'

x=Input(shape=(300,300,3))


y=Conv2D(32,3,1,activation='relu')(x)
y=Conv2D(64,3,1,activation='relu')(y)
y=Conv2D(64,3,1,activation='relu')(y)
y=MaxPooling2D((2,2))(y)
y=Dropout(0.3)(y)

y=Conv2D(128,3,1,activation='relu')(y)
y=Conv2D(128,3,1,activation='relu')(y)
y=Conv2D(256,3,1,activation='relu')(y)
y=MaxPooling2D((2,2))(y)
y=Dropout(0.3)(y)



y=Conv2D(256,3,1,activation='relu')(y)
y=Conv2D(512,3,1,activation='relu')(y)
y=MaxPooling2D((2,2))(y)
y=Dropout(0.2)(y)

z=Conv2D(16,3,1,activation='relu')(x)
z=Conv2D(32,3,1,activation='relu')(z)
z=Dense(4,activation='sigmoid',name='BBox')(z)



y=Flatten()(y)
y=Dense(100,activation='relu')(y)
y=Dense(ClassNum,activation='softmax', name='ClassLabel')(y)
model=Model(inputs=x,outputs=[y,z])

TrainImg=ImageDataGenerator(rescale=1/255).flow_from_directory(Trainpath,target_size=(300,300),classes=["Bee","Otherinsect","Wasp"],batch_size=25)
TestImg=ImageDataGenerator(rescale=1/255).flow_from_directory(TestPath,target_size=(300,300),classes=["Bee","Otherinsect","Wasp"],batch_size=25)
PredictImg=ImageDataGenerator(rescale=1/255).flow_from_directory(PredictPath,target_size=(300,300),classes=["Bee","Otherinsect","Wasp"],batch_size=12)



model.compile(Adam(),loss = {'BBox': 'mse'}, metrics=['accuracy'])


   
model.fit_generator(TrainImg,epochs=12)
   

model.save('InsectsDetection.h5')

img,label=next(PredictImg)
model=load_model('InsectsDetection.h5')
print(PredictImg.class_indices)
Plabels,Pbox=model.predict(img)
print(K.argmax(Plabels))
print(K.argmax(label))
fig=plt.figure(figsize=(15,15))

for i in range(12):
    fig.add_subplot(3,4,i+1)
    plt.imshow(img[i])

plt.show()   
