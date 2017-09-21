import numpy as np
import cv2
from camera_calib import get_calib_points, get_coefficients, undistort_image

"""
Use an image where lane lines are known to be straight
"""
def get_matrix_transform(perform_test=False):
    # load image to get the transform matrix
    print('Loading default image to calculate transform matrix...')
    input_image = cv2.imread('test_images/straight_lines1.jpg')
    img_size = (input_image.shape[0], input_image.shape[1])
    # Get camera matrix and distortion coefficients
    objpoints, imgpoints = get_calib_points()
    ret, mtx, dist, rvecs, tvecs = get_coefficients(objpoints, imgpoints, img_size)
    undistorted_image = undistort_image(input_image, mtx, dist)

    # Convert to grayscale
    # img_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # source points are found empirically
    src = np.float32([[600, 450], [230, 705], [1075, 705], [685, 450]])
    # destination points are found empirically
    dst = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])
    # cv2.line(img, pt1, pt2)
    print('Calculating transform matrix and its inverse...')
    matrix_transform = cv2.getPerspectiveTransform(src, dst)
    inv_matrix_transform = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = undistorted_image.copy()
    if perform_test:
        img_size = (undistorted_image.shape[1], undistorted_image.shape[0])
        warped = cv2.warpPerspective(undistorted_image, matrix_transform, img_size, flags=cv2.INTER_LINEAR)
        cv2.polylines(undistorted_image, [np.asarray(src, np.int32)], True, (0, 0, 255), 3)
        cv2.polylines(warped, [np.asarray(dst, np.int32)], True, (0, 0, 255), 3)
        cv2.imwrite('output_images/straight_lines1_show_lines.png', undistorted_image)
        cv2.imwrite('output_images/straight_lines1_birdeye.png', warped)

    # Return the resulting image and matrix
    return warped, matrix_transform, inv_matrix_transform

if __name__ == '__main__':
    warped, matrix_transform, _ = get_matrix_transform(perform_test=True)
