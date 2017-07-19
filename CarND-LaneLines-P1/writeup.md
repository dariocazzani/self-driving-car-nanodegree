# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

## Reflection

## 1. Pipeline Description:

    My pipeline consisted of 5 steps. 

### * Convert image to grayscale
    Colors are not important (I believe for now) in order to detect lane lines
    
### * Apply Gaussian Blur: 
    The next step includes blurring for reducing noise in edge detection, but an extra step allows us to have better control over the outcome
    
### * Edge detection using Canny algorithm

### * Masking:
    Not all image is useful for detecting lane lines as they appear to a specific area of the camera view.

### * Detect lines: 
    Using the Hough transform it is possible to associate a line to a cluster of points in the image

## Details behind how lines were detected

### Tune the Hough Transform hyperparameters
The first step was to tune the hyperparamters for the Hough transform in order to extract only lane lines.
Together with a bit of reasoning (assumption on the lenght of lane-lines, width, etc...), the approach has been mainly a trial and error.
I used the test images to test whether my tuning was good enough and if the detected lines were (mostly) just the real lane lines. 

### Find 2 lane lines 

#### First solution
The goal of the project is to find the 2 lane lines at the left and right of the camera.
At first I tried to avarage value the slope and intercept of all the lines with negative slope and positive slope respectively.
This worked well enough for the test images, but not as well for videos.
The result was pretty noisy and the output would result in lane lines moving around the video stream all the time, or at best, to be very shaky.

#### Second solution: Discard certain lines
It is appeared obvious that useful lane lines would not be horizontal, and more in general their slope would be confined to specific values.
In order to improve the output of the video lane lines detection, the avarage slope and intercept was calculated only by taking those lines whose slope is between certain limits.
This has shown huge improvements in the outcome

#### Third solution: Linear regression instead of avarage
This step is a minor improvement over the previous one. In order to find one line for the negative and positive lane lines, instead of avaraging, I find the best fir by computing a linear regression over the cluster of points defined by the Hough Transform.
See function:
##### - find_pos_and_neg_lines
##### - get_line_coeff
    
#### Final Solution: Use info from previous frame
Because results were good on the test images, but the results were alwasy "shaky" on videos, I decided to see if was possible to improve the estimation of the line position given the information on the lines at the previous frame.
This is justified by the assumption that lane lines don't "move" around too much between each frame.
Whenever the information for the previous line is available, the current position is calculated by smoothing the estimation with a first order IIR filter.
position(t_1) = alpha * position_previous_frame + beta * current_position 
See function:
##### get_line_coeff
If the information of the position of the line in the previous frame is not available, I just use the current estimation
One last note: if the estimated current position is outside the slope boundaries I mentioned above, I either ignore the current frame and do not provide the position of the lane line, or - if I have the information of the line position for the previous frame - I assume the lane line to be in the same position


![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
