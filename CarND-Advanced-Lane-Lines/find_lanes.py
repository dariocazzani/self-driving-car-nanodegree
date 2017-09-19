import numpy as np
import cv2
from camera_calib import get_calib_points, get_coefficients, undistort_image
from binary_image import get_binary_img
from perspective_transform import get_matrix_transform

def get_binary_birdeye(img):
    # Get camera matrix and distortion coefficients
    objpoints, imgpoints = get_calib_points()
    ret, mtx, dist, rvecs, tvecs = get_coefficients(objpoints, imgpoints, (720, 1280))
    undistorted_image = undistort_image(input_image, mtx, dist)
    binary_image, _ = get_binary_img(undistorted_image)
    _, M = get_matrix_transform()
    img_size = (binary_image.shape[1], binary_image.shape[0])
    warped_binary_image = cv2.warpPerspective(binary_image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped_binary_image

if __name__ == '__main__':
    test_image = 5
    input_image = cv2.imread('test_images/test{}.jpg'.format(test_image))
    binary_warped = get_binary_birdeye(input_image)

    # Take a histogram of the bottom half of the image binary birdeye image
    # NB: binary_warped is a 3 channel image - different from tutorial
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):, :, 0], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = binary_warped.copy()
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    print('midpoint: {} - leftx_base: {} - rightx_base: {}'.format(midpoint, leftx_base, rightx_base))
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to re-center window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

        # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low ,win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Draw curve on figure
    left_lane_dots = zip(list(ploty), list(left_fitx))
    right_lane_dots = zip(list(ploty), list(right_fitx))
    for l in list(left_lane_dots):
        cv2.circle(out_img, (int(l[1]), int(l[0])), 2, (0, 255, 255))
    for r in list(right_lane_dots):
        cv2.circle(out_img, (int(r[1]), int(r[0])), 2, (0, 255, 255))

    cv2.imwrite('test.png', out_img)

    # num_test_images = 6
    # for test_image in range(1, num_test_images+1):
    #     input_image = cv2.imread('test_images/test{}.jpg'.format(test_image))
    #     warped_binary_image = get_binary_birdeye(input_image)
    #     cv2.imwrite('output_images/test{}_binary_birdeye.png'.format(test_image), warped_binary_image)
