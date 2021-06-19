---
layout: single
classes: wide
title: "CNN Deep Learning on the MNIST Dataset"
header:
  teaser: /assets/images/5x3/CNN.png
  overlay_color: "#5e616c"
  overlay_image: /assets/images/1920x1080/CNN.png
  overlay_filter: 0.5
  actions:
  - label: "<i class='fas fa-arrow-alt-circle-right'></i> View in Jupyter"
    url: "https://nbviewer.jupyter.org/github/chrisman1015/Deep-Learning/blob/master/CNN%20Deep%20Learning%20on%20MNIST%20Data/CNN%20Deep%20Learning%20on%20Mnist%20Data.ipynb"
excerpt: >
  By: Christopher Hauman<br />
categories:
  - Python
  - Deep Learning
tags:
  - Python
  - Deep Learning
  - Supervised
  - CNN
---

This brief guide will cover building a simple Convolutional Neural Network with keras. This is a sequel to my more detailed guide and introduction to Neural Networks, [MLP Deep Learning on the MNIST Dataset](https://chrisman1015.github.io//python/deep%20learning/MLP-Deep-Learning-on-Mnist-Data/). This will adapt and explain the CNN example in [keras' domumentation](https://keras.io/examples/mnist_cnn/).

If you're new to CNNs, I'd highly recommend you check out [Brandon Rohrer](https://youtu.be/FmpDIaiMIeA)'s guide on them, which will give you all the theory you need to know for this implimentation guide. This type of learning also falls under the umbrella of supervised machine learning, which you can learn much more about in my guides [here](https://chrisman1015.github.io/projects/#supervised-learning).

Note: This assumes you have basic knowledge of python data science basics. If you don't, or encounter something you're not familiar with, don't worry! You can get a crash course in my guide, [Cleaning MLB Statcast Data using pandas DataFrames and seaborn Visualization](https://chrisman1015.github.io/python/statcast/Cleaning-MLB-Statcast-Data-using-pandas-Dataframes-and-seaborn-Visualization/).

***

First, I'll check to make sure keras is utilizing my GPUs and we'll import the key libraries.


```python
# get available GPUs
from keras import backend as K

K.tensorflow_backend._get_available_gpus()
```




    ['/job:localhost/replica:0/task:0/device:GPU:0',
     '/job:localhost/replica:0/task:0/device:GPU:1']




```python
# make sure GPU is available
import tensorflow as tf
tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)
```




    True




```python
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```


```python
# import libraries
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
```

Let's start by importing the data as usual:


```python
# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

A key difference between using keras for MLP neural networks and CNN neural networks is the input shape. MLP required the input be a flat image, while CNNs want the data to remain in the rectangular (in this case square) shape. 

Let's look at the shape of the X_train data:


```python
X_train.shape
```




    (60000, 28, 28)



We see the X_training data is 60000 28x28 images. For CNN input, we specifically need the input data to be in the format (batch, height, width, channels). This means we are lacking one dimension, the channel value. Channels contains the 3 RGB values for color data, but only one for grayscale images. We can fix the shape by assigning a dimension of 1 for the channel of the X_train and X_test data


```python
print('X_train before reshaping:', X_train.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)




print("X_train after reshaping:", X_train.shape )
```

    X_train before reshaping: (60000, 28, 28)
    X_train after reshaping: (60000, 28, 28, 1)
    

Now the X_train and X_test data are in the correct shape. Let's also store the input shape which we'll pass to the first CNN layer similar to the MLP example. We'll also normalize the X data and force the y data into categorical as usual.


```python
# get CNN first layer input shape
input_shape = X_train[0].shape
input_shape

# normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


num_classes = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

***
Before we build the CNN, we'll import one of the MLP models we made in the prequel for comparison:


```python
# load model
MLP_model = load_model('model_2.h5')
```


```python
MLP_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_10 (Dense)             (None, 512)               401920    
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_11 (Dense)             (None, 512)               262656    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 10)                5130      
    =================================================================
    Total params: 669,706
    Trainable params: 669,706
    Non-trainable params: 0
    _________________________________________________________________
    

This MLP model has two Dense layers of 512 neurons each, two dropout layers (which add no parameters), and a final Dense output layer of 10 neurons. Note that this led to about 700,000 total trainable parameters. From the MLP guide, we also learned this model had an accuracy score of 0.984.


Now let's build a comparable convolutional neural network: 


We intitialize our model with the Sequential function as usual, but this time we build the first two layers with Conv2D instead of Dense.

**kernel_size**
-An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
-A 3x3 kernal size means the convolutional window will be a 3x3 square.

**Flatten** 
-Flattens the output of the convolutional layer into a 2D array for Dense input:


```python
model_1 = Sequential()


model_1.add(Conv2D(512, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape, name='Conv_1'))

model_1.add(Dropout(0.2, name='Dropout_1'))

model_1.add(Conv2D(512, (3, 3), activation='relu', name='Conv_2'))

model_1.add(Dropout(0.2, name='Dropout_2'))


model_1.add(Flatten(name='Flatten'), )

model_1.add(Dense(num_classes, activation='softmax', name="Dense_output"))

model_1.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    Conv_1 (Conv2D)              (None, 26, 26, 512)       5120      
    _________________________________________________________________
    Dropout_1 (Dropout)          (None, 26, 26, 512)       0         
    _________________________________________________________________
    Conv_2 (Conv2D)              (None, 24, 24, 512)       2359808   
    _________________________________________________________________
    Dropout_2 (Dropout)          (None, 24, 24, 512)       0         
    _________________________________________________________________
    Flatten (Flatten)            (None, 294912)            0         
    _________________________________________________________________
    Dense_output (Dense)         (None, 10)                2949130   
    =================================================================
    Total params: 5,314,058
    Trainable params: 5,314,058
    Non-trainable params: 0
    _________________________________________________________________
    

This CNN model has over 7.5 times the number of trainable parameters as the MLP neural network. Let's break down the parameters in this model to see why.

Conv_1: 
512 kernels x 9 parameters each  = 4608 feature maps for the first layer. We then add 512 bias terms for each kernel. 
512 x 9 + 512 = **5120 parameters**

The Dropout and layers have no parameters.

Conv_2: 
512 kernels x  4608 feature maps + 512 bias terms = **2359808 parametes**

Flatten: 
We now take the 512 feature maps from Conv_2 and flatten it into one layer.
512 x 24 x 24 = **294912 parameters**

Dense_output: 
294912 flattened pixels x 10 output neurons + 10 bias terms = **2949130 parameters in the final layer**

***

We see that convolutional layers trigger exponential parameter growth. If we had no way to combat this, CNNs would be useless due to the extreme computational cost. Fortunately, we can use **pooling**. Pooling shrinks the size of each feature map, allowing us to use CNNs ability to detect patterns while simultaneously contolling the parameter cost.
You can read about the pooling layer [here](http://cs231n.github.io/convolutional-networks/#pool). The argument **pool_size** is a window argument, similar to **kernel_size**.


```python
model_2 = Sequential()


model_2.add(Conv2D(512, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape, name='Conv_1'))
model_2.add(MaxPooling2D(pool_size=(2, 2), name='Pool_1'))
model_2.add(Dropout(0.2, name='Dropout_1'))

model_2.add(Conv2D(512, (3, 3), activation='relu', name='Conv_2'))
model_2.add(MaxPooling2D(pool_size=(2, 2), name='Pool_2'))
model_2.add(Dropout(0.2, name='Dropout_2'))

model_2.add(Flatten(name="Flatten"))
model_2.add(Dense(num_classes, activation='softmax', name="Dense_output"))

model_2.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    Conv_1 (Conv2D)              (None, 26, 26, 512)       5120      
    _________________________________________________________________
    Pool_1 (MaxPooling2D)        (None, 13, 13, 512)       0         
    _________________________________________________________________
    Dropout_1 (Dropout)          (None, 13, 13, 512)       0         
    _________________________________________________________________
    Conv_2 (Conv2D)              (None, 11, 11, 512)       2359808   
    _________________________________________________________________
    Pool_2 (MaxPooling2D)        (None, 5, 5, 512)         0         
    _________________________________________________________________
    Dropout_2 (Dropout)          (None, 5, 5, 512)         0         
    _________________________________________________________________
    Flatten (Flatten)            (None, 12800)             0         
    _________________________________________________________________
    Dense_output (Dense)         (None, 10)                128010    
    =================================================================
    Total params: 2,492,938
    Trainable params: 2,492,938
    Non-trainable params: 0
    _________________________________________________________________
    

We see that adding these pooling layers decreased the parameters of the network by about 2/3. Let's make it smaller still, so it has the same number of total parameters as the MLP model for comparison.


```python
model_3 = Sequential()


model_3.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape, name='Conv_1'))
model_3.add(MaxPooling2D(pool_size=(2, 2), name='Pool_1'))
model_3.add(Dropout(0.2, name='Dropout_1'))

model_3.add(Conv2D(64, (3, 3), activation='relu', name='Conv_2'))
model_3.add(MaxPooling2D(pool_size=(2, 2), name='Pool_2'))
model_3.add(Dropout(0.2, name='Dropout_2'))

model_3.add(Flatten(name="Flatten"))
model_3.add(Dense(num_classes, activation='softmax', name="Dense_output"))

model_3.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    Conv_1 (Conv2D)              (None, 26, 26, 64)        640       
    _________________________________________________________________
    Pool_1 (MaxPooling2D)        (None, 13, 13, 64)        0         
    _________________________________________________________________
    Dropout_1 (Dropout)          (None, 13, 13, 64)        0         
    _________________________________________________________________
    Conv_2 (Conv2D)              (None, 11, 11, 64)        36928     
    _________________________________________________________________
    Pool_2 (MaxPooling2D)        (None, 5, 5, 64)          0         
    _________________________________________________________________
    Dropout_2 (Dropout)          (None, 5, 5, 64)          0         
    _________________________________________________________________
    Flatten (Flatten)            (None, 1600)              0         
    _________________________________________________________________
    Dense_output (Dense)         (None, 10)                16010     
    =================================================================
    Total params: 53,578
    Trainable params: 53,578
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# compile the model
model_3.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
```

We'll fit the model with an early stopping monitor as well.


```python
# initialize early stopping monitor
early_stopping_monitor = EarlyStopping(patience=3)

batch_size = 128
epochs = 12

model_3.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks = [early_stopping_monitor],
          validation_split=0.2,
          verbose=1)
```

    Train on 48000 samples, validate on 12000 samples
    Epoch 1/12
    48000/48000 [==============================] - 35s 722us/step - loss: 0.0181 - acc: 0.9936 - val_loss: 0.0118 - val_acc: 0.9966
    Epoch 2/12
    48000/48000 [==============================] - 36s 757us/step - loss: 0.0169 - acc: 0.9945 - val_loss: 0.0132 - val_acc: 0.9962
    Epoch 3/12
    48000/48000 [==============================] - 36s 747us/step - loss: 0.0149 - acc: 0.9952 - val_loss: 0.0139 - val_acc: 0.9961
    Epoch 4/12
    48000/48000 [==============================] - 35s 734us/step - loss: 0.0145 - acc: 0.9951 - val_loss: 0.0166 - val_acc: 0.9952
    




    <keras.callbacks.History at 0x23799b46048>




```python
score = model_3.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Test loss: 0.027888490935719164
    Test accuracy: 0.9932
    

Look at how high the accuracy is! For image classfication, and it achieved that value on the first epoch. CNNs are an incredibly useful tool.
