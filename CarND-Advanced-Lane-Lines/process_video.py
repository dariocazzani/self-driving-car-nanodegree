from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
import cv2
import glob
import numpy as np
from binary_image import get_binary_img
from find_lanes import get_lanes_full_search, get_lanes_from_previous
from draw_lanes import draw_lane
import sys
def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[{}] {}{} ...{}\r'.format(bar, percents, '%', suffix))
    sys.stdout.flush()

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

if __name__ == '__main__':
    """
    PARAMETERS, UTILS AND PATHS
    """
    clip = VideoFileClip("project_video.mp4")
    frames = int(clip.fps * clip.duration)
    image_folder = "frames/"
    video_file = 'processed_video.mp4'

    """
    1. Camera Calibration parameters
    """
    from camera_calib import get_calib_points, get_coefficients, undistort_image
    objpoints, imgpoints = get_calib_points()
    img_road = cv2.imread('test_images/straight_lines1.jpg')
    img_road_size = (img_road.shape[1], img_road.shape[0])
    ret, mtx, dist, rvecs, tvecs = get_coefficients(objpoints, imgpoints, img_road_size)

    """
    2. Perspective Transformation parameters
    """
    from perspective_transform import get_matrix_transform
    _, matrix_transform, inv_matrix_transform = get_matrix_transform()

    """
    3. Loop over all frames and:
        - Undistort frame
        - Get thresholded binary image for lane detection
        - Warp image to get birdeye view of the street
    """
    print('Processing video...')
    for idx, frame in enumerate(clip.iter_frames()):
        progress(idx+1, frames)
        # Unfistort frame
        undistorted_frame = undistort_image(frame, mtx, dist)

        # Get thresholded binary image for lane detection
        binary_frame, _ = get_binary_img(undistorted_frame)

        # Warp image to get birdeye view of the street
        img_size = (binary_frame.shape[1], binary_frame.shape[0])
        warped = cv2.warpPerspective(binary_frame, matrix_transform, img_size, flags=cv2.INTER_LINEAR)

        # Draw lanes
        img_with_lanes = draw_lane(warped, undistorted_frame)

        cv2.imwrite('{}frame_{:010d}.jpg'.format(image_folder, idx), img_with_lanes)
    print('')
