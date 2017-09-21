import numpy as np
import cv2
from camera_calib import get_calib_points, get_coefficients, undistort_image
from binary_image import get_binary_img
from perspective_transform import get_matrix_transform
from find_lanes import get_lanes_full_search, get_binary_birdeye

def draw_lane(input_image):
    img_size = (input_image.shape[0], input_image.shape[1])
    # Get camera matrix and distortion coefficients
    objpoints, imgpoints = get_calib_points()
    # get perspective transformation matrix
    _, M, Minv = get_matrix_transform()

    ret, mtx, dist, rvecs, tvecs = get_coefficients(objpoints, imgpoints, img_size)
    undist = undistort_image(input_image, mtx, dist)

    warped = get_binary_birdeye(input_image)

    lefty, leftx, righty, rightx, out_img = get_lanes_full_search(warped)
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

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

if __name__ == '__main__':
    num_test_images = 6
    for test_image in range(1, num_test_images+1):
        input_image = cv2.imread('test_images/test{}.jpg'.format(test_image))
        result = draw_lane(input_image)
        cv2.imwrite('output_images/test{}_with_drawn_lanes.jpg'.format(test_image), result)
