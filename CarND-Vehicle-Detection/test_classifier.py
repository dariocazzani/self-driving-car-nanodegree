import pickle
import matplotlib.image as mpimg
from features import get_features
from training_data import get_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

if __name__ == '__main__':
    # load classifier
    with open('clf.pkl', 'rb') as fid:
        clf = pickle.load(fid)
    # load scaler
    with open('scaler.pkl', 'rb') as fid:
        X_scaler = pickle.load(fid)

    features_car = get_features('vehicles/GTI_Far/image0004.png')
    features_noncar = get_features('non-vehicles/GTI/image1.png')

    features_car_scaled = X_scaler.transform(features_car)
    features_noncar_scaled = X_scaler.transform(features_noncar)

    prediction = clf.predict(features_car_scaled)
    if prediction == 1:
        print('Hurray!')
    prediction = clf.predict(features_noncar_scaled)
    if prediction == 0:
        print('Hurray!')

    X_train, y_train, X_test, y_test = get_data()
    scaled_X_test = X_scaler.transform(X_test)
    predictions = clf.predict(scaled_X_test)
    print('Accuracy on Test Set: {:.2f}%'.format(accuracy_score(y_test, predictions)))

    print("\nDetailed classification report:")
    print(classification_report(y_test, predictions))
