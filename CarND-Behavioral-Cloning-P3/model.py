from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.layers import Convolution2D

import os
import csv
import cv2
import numpy as np
import sklearn
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
    samples = []
    source = 'Data'
    with open('{}/driving_log.csv'.format(source)) as csvfile:
        reader = csv.reader(csvfile)
        for line in list(reader)[1:]:
            samples.append(line)

    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)


    def generator(samples, batch_size=32):
        num_samples = len(samples)
        while 1: # Loop forever so the generator never terminates
            random.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    name = '{}/IMG/{}'.format(source, batch_sample[0].split('/')[-1])
                    center_image = cv2.imread(name)
                    rgb_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    center_angle = float(batch_sample[3])
                    if np.random.randint(2) == 1:
                        images.append(rgb_image)
                        angles.append(center_angle)
                    else:
                        images.append(cv2.flip(rgb_image, 1))
                        angles.append((-1) * center_angle)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)
    print(train_generator.__next__()[0].shape)

    ch, row, col = 3, 80, 320  # Trimmed image format

    model = Sequential()
    model.add(Lambda(lambda x: x / 255. -0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, W_regularizer=l2(0.001), activation='relu'))
    model.add(Dense(50, W_regularizer=l2(0.001), activation='relu'))
    model.add(Dense(10, W_regularizer=l2(0.001), activation='relu'))

    # Add a fully connected output layer
    model.add(Dense(1))

    print('Training...')
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    history_object = model.fit_generator(train_generator,
              samples_per_epoch=len(train_samples),
              validation_data=validation_generator,
              nb_val_samples=len(validation_samples),
              nb_epoch=20)

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    # Bug fix: Keras cant delete session
    import gc; gc.collect()

    model.save('model.h5')
