import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_binary_img(img, s_thresh=(120, 255), sx_thresh=(20, 255),l_thresh=(40,255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

     # Threshold lightness
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    # Stack each channel
    channels = 255*np.dstack(( l_binary, sxbinary, s_binary)).astype('uint8')

    # Combine the three binary thresholds
    binary = np.zeros_like(sxbinary)
    binary[((l_binary == 1) & (s_binary == 1) | (sxbinary==1))] = 1
    binary = 255*np.dstack((binary,binary,binary)).astype('uint8')

    return binary, channels

if __name__ == '__main__':
    num_test_images = 6
    for test_image in range(1, num_test_images+1):
        image = cv2.imread('test_images/test{}.jpg'.format(test_image))
        combined_binary, color_binary = get_binary_img(image)
        cv2.imwrite('output_images/test{}_binary.png'.format(test_image), combined_binary)
        cv2.imwrite('output_images/test{}_color_binary.png'.format(test_image), color_binary)
