import csv
import cv2
import numpy as np

def load_data(source):
    lines = []
    with open('{}/driving_log.csv'.format(source)) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    print('Loading data...')
    for line in lines[1:]:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = '{}/IMG/'.format(source) + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurements.append(float(line[3]))
        images.append(cv2.flip(image, 1))
        measurements.append(float(line[3])*(-1))

    X_train = np.asarray(images)
    y_train = np.asarray(measurements)
    return X_train, y_train
