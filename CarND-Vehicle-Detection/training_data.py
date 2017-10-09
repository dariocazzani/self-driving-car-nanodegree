from features import get_features
import glob, os
import numpy as np
from sklearn.utils import shuffle

def get_data():
    if os.path.exists('X_train.npy'):
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')

    else:
        cars = glob.glob('vehicles/**/*.png')
        non_cars = glob.glob('non-vehicles/**/*.png')

        # Compute features and labels for training data
        car_features = []
        non_car_features = []
        y = []
        for idx, car in enumerate(cars):
            if idx % 1000 == 0:
                print('extracting features for car: {}'.format(idx))
            car_features.append(get_features(car))
            y.append(1)
        for idx, noncar in enumerate(non_cars):
            if idx % 1000 == 0:
                print('extracting features for non car: {}'.format(idx))
            non_car_features.append(get_features(noncar))
            y.append(0)

        car_features = np.asarray(car_features)
        non_car_features = np.asarray(non_car_features)
        y = np.asarray(y)
        X = np.squeeze(np.concatenate((car_features, non_car_features)))

        # Shuffle features and labels in a consistent way
        X, y = shuffle(X, y)

        # Split training and test data
        train_portion = 0.75
        tot_training_samples = int(len(X) * train_portion)
        X_train = X[:tot_training_samples]
        y_train = y[:tot_training_samples]
        X_test = X[tot_training_samples:]
        y_test = y[tot_training_samples:]

        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)
        np.save('X_test.npy', X_test)
        np.save('y_test.npy', y_test)

    print('\nNumber of training samples: {}'.format(X_train.shape[0]))
    print('Number of test samples: {}'.format(X_test.shape[0]))
    print('Number of positive samples: {}'.format(np.sum(y_test) + np.sum(y_train)))

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data()
