import numpy as np

from training_data import get_data
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn import svm

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data()

    # compute scaler
    X_scaler = StandardScaler()
    X_scaler.fit(X_train)
    scaled_X_train = X_scaler.transform(X_train)
    scaled_X_test = X_scaler.transform(X_test)

    # hyperparameters
    # parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
    #               {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    #
    # parameters = [{'kernel': ['rbf'], 'gamma': [1e-3], 'C': [1]}]
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

    # Random search of hyperparameters
    print('(Optional - Performing hyperparameters search and) Training classifier')
    # svr = svm.SVC()
    clf = svm.SVC()
    # clf = GridSearchCV(svr, parameters)
    clf.fit(scaled_X_train, y_train)
    # print("Best parameters set found on development set:")
    # print(clf.best_params_)
    # print("Grid scores on development set:")
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))

    # Save classifier
    import pickle
    # save the classifier
    with open('clf.pkl', 'wb') as fid:
        pickle.dump(clf, fid)
    with open('scaler.pkl', 'wb') as fid:
        pickle.dump(X_scaler, fid)

    # # load it again
    # with open('clf.pkl', 'rb') as fid:
    #     clf = cPickle.load(fid)
