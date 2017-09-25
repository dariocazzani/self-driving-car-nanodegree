import numpy as np
import cv2
from camera_calib import get_calib_points, get_coefficients, undistort_image
from binary_image import get_binary_img
from find_lanes import get_lanes_full_search, get_binary_birdeye

def get_curvatures_and_offset(lefty, leftx, righty, rightx, binary_warped):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    y_eval = np.max(ploty)

    # Fit a second order polynomial to each lane
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    # Generate points
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

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
    lane_center = (right_fitx[-1] + left_fitx[-1]) / 2.
    center_offset_pixels = camera_position - lane_center
    center_offset_meters = xm_per_pix * center_offset_pixels

    return left_curverad, right_curverad, center_offset_meters, center_offset_pixels

def get_curvatures_in_pixels(left_fit, right_fit, binary_warped):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    y_eval = np.max(ploty)

    # Calculate the new radii of curvature
    left_curverad_pixels = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad_pixels = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    return left_curverad_pixels, right_curverad_pixels

if __name__ == '__main__':
    num_test_images = 6
    for test_image in range(1, num_test_images+1):
        input_image = cv2.imread('test_images/test{}.jpg'.format(test_image))
        binary_warped = get_binary_birdeye(input_image)
        lefty, leftx, righty, rightx, out_img = get_lanes_full_search(binary_warped)
        out = get_curvatures_and_offset(lefty, leftx, righty, rightx, binary_warped)
        left_curverad, right_curverad, center_offset_meters, center_offset_pixels = out
        print('left_curverad {:.2f} meters - right_curverad: {:.2f} meters'.format(left_curverad, right_curverad))
        if center_offset_meters < 0:
            print('offset: {:.3f} meters to the left'.format(abs(center_offset_meters)))
        else:
            print('offset: {:.3f} meters to the right'.format(abs(center_offset_meters)))
        # Fit a second order polynomial to each lane
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        out = get_curvatures_in_pixels(left_fit, right_fit, binary_warped)
        left_curverad_pixels, right_curverad_pixels = out
        print('left_curverad_pixels {:.2f} pixels - right_curverad_pixels: {:.2f} pixels'.format(left_curverad_pixels, right_curverad_pixels))
        if center_offset_pixels < 0:
            print('offset: {:.3f} pixels to the left'.format(abs(center_offset_pixels)))
        else:
            print('offset: {:.3f} pixels to the right'.format(abs(center_offset_pixels)))

        print('Pixel to meters ration in radii: {}'.format(left_curverad_pixels/left_curverad))
        print('Pixel to meters ration in radii: {}'.format(right_curverad_pixels/right_curverad))

        print('')
