from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
import matplotlib.image as mpimg
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from sliding_window_classify import find_cars
from heat_map import *
import glob
import sys

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[{}] {}{} ...{}\r'.format(bar, percents, '%', suffix))
    sys.stdout.flush()

if __name__ == '__main__':
    """
    PARAMETERS, UTILS AND PATHS
    """
    clip = VideoFileClip("project_video.mp4")
    frames = int(clip.fps * clip.duration)
    image_folder = "frames/"
    video_file = 'processed_video.mp4'

    # Hyperparams
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (16, 16)
    hist_bins=32

    ystart = 400
    ystop = 650
    scales = [1.0, 1.3, 1.7]

    THRESHOLD = 5

    # load classifier
    print('Loading Classifier')
    with open('clf.pkl', 'rb') as fid:
        clf = pickle.load(fid)
    print('Loading Scaler')
    # load scaler
    with open('scaler.pkl', 'rb') as fid:
        X_scaler = pickle.load(fid)

    print('Processing video...')
    for idx, img in enumerate(clip.iter_frames()):
        progress(idx+1, frames)

        _, box_list = find_cars(img, ystart, ystop, scales, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

        if idx == 0:
            heat = np.zeros_like(img[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat, box_list)
        heat = remove_old_heat(heat)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, THRESHOLD)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        mpimg.imsave('frames/test{}_detections.png'.format(idx+1), draw_img)
