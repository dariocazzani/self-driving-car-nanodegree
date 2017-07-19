# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Pipeline Description:

    My pipeline consisted of 5 steps. 

#### * Convert image to grayscale
    Colors are not important (I believe for now) in order to detect lane lines
    
#### * Apply Gaussian Blur: 
    Step number 3 includes blurring for reducing noise in edge detection, but an extra step allows us to have better control over the outcome
    
#### * Edge detection using Canny algorithm

#### * Masking:
    Not all image is useful for detecting lane lines as they appear to a specific area of the camera view.

#### * Detect lines: 
    Using the Hough transform it is possible to associate a line to a cluster of points in the image

### Details behind how lines were detected

    

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
