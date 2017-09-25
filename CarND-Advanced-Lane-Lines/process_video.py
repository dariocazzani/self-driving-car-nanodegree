from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
import cv2
import glob
import numpy as np
from binary_image import get_binary_img
from find_lanes import get_lanes_full_search, get_lanes_from_previous
from draw_lanes import draw_lane
from curvature import get_curvatures_in_pixels, get_curvatures_and_offset
import sys

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[{}] {}{} ...{}\r'.format(bar, percents, '%', suffix))
    sys.stdout.flush()

def draw(warped, left_fit, right_fit, Minv, undist):
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    color_warp = np.zeros_like(warped).astype(np.uint8)
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result

def add_text(img, left_curverad, right_curverad, center_offset_meters):
    # Add info about radius and offset
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = 'left_curverad {:5.2f} meters - right_curverad: {:5.2f}'.format(left_curverad, right_curverad)
    if center_offset_meters < 0:
        text2 = 'offset: {:2.3f} meters to the left'.format(abs(center_offset_meters))
    else:
        text2 = 'offset: {:2.3f} meters to the right'.format(abs(center_offset_meters))

    cv2.putText(img, text1, (50, 30), font, 1, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(img, text2, (50, 70), font, 1, (0,0,255), 1, cv2.LINE_AA)

    return result

if __name__ == '__main__':
    """
    PARAMETERS, UTILS AND PATHS
    """
    clip = VideoFileClip("project_video.mp4")
    frames = int(clip.fps * clip.duration)
    image_folder = "frames/"
    video_file = 'processed_video.mp4'
    # smoothing params
    alpha = 0.1
    beta = 0.9
    # pixel to meters conversion
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    pxl_to_meters_radius_ratio = 3.05

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
        - Detect lanes and smooth between frames
        - Fit a second order polynomial to each lane
        - Calculate radii and vehicle offset
        - Draw lanes and add text
        - Save frame

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

        # Detect lanes and smooth between frames
        if idx == 0:
            lefty, leftx, righty, rightx, _ = get_lanes_full_search(warped)
            # Fit a second order polynomial to each lane
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            _, _, _, center_offset_pixels = get_curvatures_and_offset(lefty, leftx, righty, rightx, warped)
        else:
            lefty, leftx, righty, rightx = get_lanes_from_previous(warped, left_fit, right_fit)
            # Fit a second order polynomial to each lane
            next_left_fit = np.polyfit(lefty, leftx, 2)
            next_right_fit = np.polyfit(righty, rightx, 2)
            _, _, _, next_center_offset_pixels = get_curvatures_and_offset(lefty, leftx, righty, rightx, warped)

            # smooth detected lanes
            left_fit = alpha * left_fit + beta * next_left_fit
            right_fit = alpha * right_fit + beta * next_right_fit

        # Calculate radii and vehicle offset in pixels
        curvatures = get_curvatures_in_pixels(left_fit, right_fit, warped)
        left_curverad_pixels, right_curverad_pixels = curvatures

        # smooth center offset
        if idx > 0:
            center_offset_pixels = alpha * center_offset_pixels + beta * next_center_offset_pixels

        # Convert to meters
        center_offset_meters = xm_per_pix * center_offset_pixels
        left_curverad = left_curverad_pixels / pxl_to_meters_radius_ratio
        right_curverad = right_curverad_pixels / pxl_to_meters_radius_ratio

        # Draw_lanes
        result = draw(warped, left_fit, right_fit, inv_matrix_transform, undistorted_frame)

        # Add text
        result = add_text(result, left_curverad, right_curverad, center_offset_meters)

        # save frame
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        cv2.imwrite('{}frame_{:010d}.jpg'.format(image_folder, idx), rgb)
    print('')
