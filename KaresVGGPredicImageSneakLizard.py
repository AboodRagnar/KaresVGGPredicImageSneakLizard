import keras
from keras.layers import Activation
from keras.models import Sequential
from  keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
import numpy as np
import warnings
from keras.models import load_model
warnings.simplefilter(action='ignore', category=FutureWarning)
Model=Sequential()
train_batches = ImageDataGenerator().flow_from_directory('T3/Images/SnakeLisard/Train', target_size=(224,224),
    classes=['Snake', 'Lizard'], batch_size=3)
test_batches = ImageDataGenerator().flow_from_directory('T3/Images/SnakeLisard/Test', target_size=(224,224),
    classes=['Snake', 'Lizard'], batch_size=3)
pridect_batches = ImageDataGenerator().flow_from_directory('T3/Images/SnakeLisard/predict', target_size=(224,224),
    classes=['Snake', 'Lizard'],batch_size=5)


# for ly in VGG16().layers[:-1]:
#     Model.add(ly)



# for layer in Model.layers:
#     layer.trainable=False



# Model.add(Dense(2,activation='softmax'))
# Model.compile(Adam(0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
# Model.fit_generator(train_batches,steps_per_epoch=20,validation_data=test_batches,validation_steps=10,epochs=10,verbose=2)
#Model.save('LizardSnakeClassifier.h5')
#print(Model.predict_generator(pridect_batches,steps=4))



#image=load_img('T3/Images/SnakeLisard/Predict/Snake/ss161.jpg',target_size=(224,224))
Model=load_model('LizardSnakeClassifier.h5')
print(Model.predict_generator(pridect_batches,steps=4))
print(pridect_batches.class_indices)



def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

imgs, labels = next(pridect_batches)
plots(imgs, titles=labels)
plt.show()
