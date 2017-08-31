from load_data import load_data
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
from keras.layers import Convolution2D

if __name__ == '__main__':
    X_train, y_train = load_data('data_example')
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. -0.5, input_shape=(160, 320, 3)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1))

    print('Training...')
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

    # Bug fix: Keras cant delete session
    import gc; gc.collect()

    model.save('model.h5')
