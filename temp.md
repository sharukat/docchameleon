```python
# import required modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

# Load in the data
cifar10 = tf.keras.datasets.cifar10

# Distribute it to train and test set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Reduce pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# flatten the label values
y_train, y_test = y_train.flatten(), y_test.flatten()

# number of classes
K = len(set(y_train))
# calculate total number of classes for output layer
print("number of classes:", K)

# Build the model using the functional API
# input layer
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
# Hidden layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
# last hidden layer i.e.. output layer
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

# saving and loading the .h5 model
# save model
model.save('gfgModel.h5')
print('Model Saved!')

# load model
savedModel = load_model('gfgModel.h5')

# saving and loading the model weights
# save model
model.save_weights('gfgModelWeights')
print('Model Saved!')

# load model
savedModel = model.load_weights('gfgModelWeights')
print('Model Loaded!')

# saving and loading the .h5 model
# save model
model.save_weights('gfgModelWeights.h5')
print('Model Saved!')

# load model
savedModel = model.load_weights('gfgModelWeights.h5')
print('Model Loaded!')
```

This code demonstrates how to save and load TensorFlow models in Python in different formats:
- Saving and loading the entire model in .h5 format using `model.save()` and `load_model()`
- Saving and loading just the model weights using `model.save_weights()` and `model.load_weights()`
- Saving the weights in .h5 format as well using `model.save_weights()`

The key points are:
- The `save()` method saves the model architecture, weights, and optimizer state
- `save_weights()` only saves the model weights
- Models are loaded with `load_model()` and weights with `load_weights