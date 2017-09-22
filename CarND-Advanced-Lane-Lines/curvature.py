import numpy as np
import cv2
from camera_calib import get_calib_points, get_coefficients, undistort_image
from binary_image import get_binary_img
from find_lanes import get_lanes_full_search, get_binary_birdeye

def get_curvatures_and_offset(lefty, leftx, righty, rightx, binary_warped):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Calcualate vehicle offset
    camera_position = binary_warped.shape[1] / 2.
    lane_center = (rightx[719] + leftx[719]) / 2.
    center_offset_pixels = abs(camera_position - lane_center)
    center_offset_meters = xm_per_pix * center_offset_pixels

    return left_curverad, right_curverad, center_offset_meters

if __name__ == '__main__':
    num_test_images = 6
    for test_image in range(1, num_test_images+1):
        input_image = cv2.imread('test_images/test{}.jpg'.format(test_image))
        binary_warped = get_binary_birdeye(input_image)
        lefty, leftx, righty, rightx, out_img = get_lanes_full_search(binary_warped)
        left_curverad, right_curverad, center_offset_meters = get_curvatures_and_offset(lefty, leftx, righty, rightx, binary_warped)
        print('left_curverad {:.2f} meters - right_curverad: {:.2f}'.format(left_curverad, right_curverad))
        print('offset: {:.3f} meters'.format(center_offset_meters))
