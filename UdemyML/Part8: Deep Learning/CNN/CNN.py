#Convolutional Neural Network:

#import libraries:
from keras.models import Sequential
#used to initialize the neural network
from keras.layers import Convolution2D
#used convolution step (2D -> image, 3D -> video)
from keras.layers import MaxPooling2D
#used for pooling step
from keras.layers import Flatten
#used for flattening 
#(transform pooled features into a large vector of features which will be feeded into the fully connected NN)
from keras.layers import Dense
#used for the fully connected NN (traditional ANN)

#initialize the CNN
classifier = Sequential()

#Step1: Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape= (64, 64, 3), activation='relu'))
#32 is nb of features, we used a 3*3 feature detector matrix on a 64*64 colored img

#Step2: Pooling 
classifier.add(MaxPooling2D(pool_size=(2,2)))
#here we use a 2*2 matrix for pooling (bacisally here we extract import information from the convolued matrix which we obtain from convolution step)

#add second convolutional hidden layer:
classifier.add(Convolution2D(32, (3, 3), activation='relu')) 
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step3: Flattening:
classifier.add(Flatten())

#Step4: Full Connection:
#hidden layer:
#it is a commun practice to use ReLU function for hidden layers and Sigmoid for output layer 
classifier.add(Dense(units=128, activation = 'relu'))
#output layer:
classifier.add(Dense(units=1, activation='sigmoid'))

#Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Image Preprocessing:
from keras.preprocessing.image import ImageDataGenerator
#this will guarantee image augmentation via multiple operation such as zoom, rotation...
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)



