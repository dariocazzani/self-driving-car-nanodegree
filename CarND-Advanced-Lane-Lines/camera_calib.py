import numpy as np
import cv2
import glob

"""
extract object points and image points for camera calibration
"""
def get_calib_points():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')
    # Step through the list and search for chessboard corners
    print('Finding corners and adding object and image points..')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            # cv2.imwrite('img_{}.png'.format(idx), img)
            # cv2.waitKey(500)

    cv2.destroyAllWindows()
    return objpoints, imgpoints

def get_coefficients(objpoints, imgpoints, img_size):
    # Do camera calibration given object points and image points
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return cv2.calibrateCamera(objpoints, imgpoints, img_size, None ,None)

def undistort_image(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

if __name__ == '__main__':
    # Test undistortion on 2 images

    # Load images and compute their size
    img_chessboard = cv2.imread('camera_cal/calibration1.jpg')
    img_chessboard_size = (img_chessboard.shape[1], img_chessboard.shape[0])
    img_road = cv2.imread('test_images/straight_lines1.jpg')
    img_road_size = (img_road.shape[1], img_road.shape[0])

    # Get calibration points (1 time op.)
    objpoints, imgpoints = get_calib_points()

    # undistort first image and save
    ret, mtx, dist, rvecs, tvecs = get_coefficients(objpoints, imgpoints, img_chessboard_size)
    undistorted_chessboard = undistort_image(img_chessboard, mtx, dist)
    cv2.imwrite('output_images/calibration1.png', img_chessboard)
    cv2.imwrite('output_images/calibration1_undist.png', undistorted_chessboard)

    # undistort second image and save
    ret, mtx, dist, rvecs, tvecs = get_coefficients(objpoints, imgpoints, img_road_size)
    undistorted_road = undistort_image(img_road, mtx, dist)
    cv2.imwrite('output_images/straight_lines1.png', img_road)
    cv2.imwrite('output_images/straight_lines1_undist.png', undistorted_road)
