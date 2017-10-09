# load data
# extract features
# shuffle
# scale
# train and parameters search
# save classifier

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import glob

def bin_spatial(img, size=(16, 16)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                              visualise=True, feature_vector=False)
    if vis == True:
        return features, hog_image
    return features

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def get_features(img_path, verbose=False):
    # Hyperparams
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (16, 16)
    hist_bins=32


    img = mpimg.imread(img_path)
    # if np.max(img) > 1:
    #     img = img.astype(np.float32)/255
    hls = convert_color(img, conv='RGB2HLS')

    ch1 = hls[:,:,0]
    ch2 = hls[:,:,1]
    ch3 = hls[:,:,2]
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
    hog_features = np.hstack((hog1, hog1, hog1))

    # Get color features
    spatial_features = bin_spatial(img, size=spatial_size)
    hist_features = color_hist(img, nbins=hist_bins)
    if verbose:
        print('hog_features: {}'.format(hog_features.shape))
        print('spatial_features: {}'.format(spatial_features.shape))
        print('spatial_features: {}'.format(spatial_features.shape))

    return np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)

if __name__ == '__main__':
    features = get_features('vehicles/GTI_Far/image0000.png', verbose=True)
    print('features: {}'.format(features.shape))
