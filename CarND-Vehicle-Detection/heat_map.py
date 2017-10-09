import matplotlib.image as mpimg
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from sliding_window_classify import find_cars
import glob

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, yxt2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    heatmap[heatmap >= 10] = 10
    # Return thresholded map
    return heatmap

def remove_old_heat(heatmap):
    heatmap[heatmap >= 1] -= 1
    return heatmap
    
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # if bounding box is too "vertical" or "horizontal" it's probably not a car
        height = float(np.abs(np.max(nonzeroy) - np.min(nonzeroy)))
        width = float(np.abs(np.max(nonzerox) - np.min(nonzerox)))
        if height / np.maximum(width, 0.001) > 2 or width / np.maximum(height, 0.001) > 2:
            continue
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

if __name__ == '__main__':

    # Hyperparams
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (16, 16)
    hist_bins=32

    ystart = 400
    ystop = 650
    scales = [1.0, 1.3, 1.7]

    # load classifier
    print('Loading Classifier')
    with open('clf.pkl', 'rb') as fid:
        clf = pickle.load(fid)
    print('Loading Scaler')
    # load scaler
    with open('scaler.pkl', 'rb') as fid:
        X_scaler = pickle.load(fid)

    test_images = glob.glob('test_images/*jpg')
    for idx, image in enumerate(test_images):
        print('Looking for cars in test image: test_images/test{}.jpg'.format(idx+1))
        img = mpimg.imread(image)
        _, box_list = find_cars(img, ystart, ystop, scales, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat, box_list)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 2)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        mpimg.imsave('output_images/test{}_heat_map_test.png'.format(idx+1), labels[0], cmap='hot')

        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        mpimg.imsave('output_images/test{}_detections.png'.format(idx+1), draw_img)
