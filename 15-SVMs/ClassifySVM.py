def classify(features_train, labels_train):
    C = 1000.0
    gamma = 1.0
    from sklearn.svm import SVC
    clf = SVC(C=C, gamma=gamma)
    clf.fit(features_train, labels_train)
    return clf
