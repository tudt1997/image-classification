# importing augmentation library
# Initial Setup for Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

def image_model():
    model = Sequential()
    # Cropping image
    model.add(Lambda(lambda imgs: imgs[:, :, :, :], input_shape=(32, 32, 3)))
    # Normalise the image - center the mean at 0
    # model.add(Lambda(lambda imgs: (imgs / 255.0) - 0.5))
    # model.add(Lambda(resize))

    # We have a series of 3 5x5 convolutional layers with a stride of 2x2
    model.add(Conv2D(32, 5, strides=(1, 1), activation='elu'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    
    # This is then followed by 2 3x3 convolutional layers with a 1x1 stride
    model.add(Conv2D(64, 3, strides=(1, 1), activation='elu')) 
    model.add(MaxPooling2D())
    model.add(BatchNormalization())

    model.add(Conv2D(128, 3, strides=(1, 1), activation='elu'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    
    # model.add(Dropout(rate=0.5))

    # Flattening the output of last convolutional layer before entering fully connected phase
    model.add(Flatten())
    
    # Fully connected layers
    # model.add(Dense(1164, activation='elu'))
    # model.add(BatchNormalization())
    
    model.add(Dense(512, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))

    model.add(Dense(256, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    
    # Output layer
    model.add(Dense(3, activation='softmax'))
    
    # model.compile(loss = "mean_squared_error", optimizer = Adam(lr = 0.001))
    return model