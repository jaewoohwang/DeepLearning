from google.colab import drive
drive.mount('/content/drive')


# Load Package
import os
import sys
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LayerNormalization
# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# Parameters
number_of_classes = 10
batch_size = 32
epochs = 150
dropout_rate = 0.2
learning_rate = 0.001
l1_rate = 0.01
momentum = 0.9
decay = 1e-6


# Load cifar10 dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()


# One-hot encoding
trainY = to_categorical(trainY, number_of_classes)
testY = to_categorical(testY, number_of_classes)


# Normalize given image dataset
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0


# Define model

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(LayerNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(LayerNormalization())
model.add(AveragePooling2D((2, 2)))
model.add(Dropout(dropout_rate))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(LayerNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(LayerNormalization())
model.add(AveragePooling2D((2, 2)))
model.add(Dropout(dropout_rate))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(LayerNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(LayerNormalization())
model.add(AveragePooling2D((2, 2)))
model.add(Dropout(dropout_rate))

model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(LayerNormalization())
model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(LayerNormalization())
model.add(AveragePooling2D((2, 2)))
model.add(Dropout(dropout_rate))

model.add(Flatten())
model.add(Dense(256, activation='relu', 
                kernel_initializer='he_uniform',
                kernel_regularizer = l1(l1_rate)
                ))
model.add(LayerNormalization())
model.add(Dropout(dropout_rate))
model.add(Dense(number_of_classes, 
                activation='softmax'))

optimizer = SGD(learning_rate = learning_rate, momentum = momentum, decay = decay)
model.compile(optimizer = optimizer, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# Set Directory
save_dir = os.path.join(os.getcwd(), 'drive/MyDrive/Colab Notebooks')

model_name = 'model.{epoch:02d}.h5' 
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=False)

callbacks = checkpoint


hist = model.fit(trainX, trainY,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(testX, testY),
          shuffle=True,
          callbacks=callbacks)


with open('/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)
history = pickle.load(open('/trainHistoryDict', "rb"))


#Plot loss and accuracy
def Plot_history(history):
  plt.subplot(211)
  plt.plot(hist.history['loss'], color = 'b', label = 'train')
  plt.plot(hist.history['val_loss'], color = 'g', label = 'test')
  plt.ylabel('Loss')
  plt.legend()

  plt.subplot(212)
  plt.plot(hist.history['accuracy'], color = 'r', label = 'train')
  plt.plot(hist.history['val_accuracy'], color = 'black', label = 'test')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()


Plot_history(history)


# print the accuracy of trained model
scores = model.evaluate(testX, testY, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
