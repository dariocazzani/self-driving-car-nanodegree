import matplotlib.image as mpimg
import numpy as np
import pickle
import cv2
from features import get_hog_features, bin_spatial, color_hist, convert_color
import glob

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scales, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2HLS')
    ctrans_tosearch = img_tosearch
    counter = 0
    for scale_idx, scale in enumerate(scales):
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient*cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                counter+=1
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                # mpimg.imsave('output_images/subimg{}.png'.format(counter), subimg)


                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = clf.predict(test_features)

                if test_prediction == 1:
                    color=(255,0,0)
                    if scale_idx == 0:
                        color = (0,0,255)
                    elif scale_idx == 1:
                        color = (0,255,0)
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),color=color, thickness=6)

    return draw_img

if __name__ == '__main__':

    # Hyperparams
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (16, 16)
    hist_bins=32

    ystart = 400
    ystop = 656
    scales = [1., 1.5, 2.5]

    # load classifier
    print('Loading Classifier')
    with open('clf.pkl', 'rb') as fid:
        clf = pickle.load(fid)
    print('Loading Scaler')
    # load scaler
    with open('scaler.pkl', 'rb') as fid:
        X_scaler = pickle.load(fid)

    test_images = glob.glob('test_images/*jpg')
    for idx, image in enumerate(test_images):
        print('Looking for cars in test image: test_images/test{}.jpg'.format(idx+1))
        img = mpimg.imread(image)
        out_img = find_cars(img, ystart, ystop, scales, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        mpimg.imsave('output_images/test{}_sliding_window_test.png'.format(idx+1), out_img)
