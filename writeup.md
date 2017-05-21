##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/example_car.png
[image2]: ./output_images/example_notcar.png
[image3]: ./output_images/example_car_for_hog.png
[image4]: ./output_images/example_hog.png
[image5]: ./output_images/example_notcar_for_hog.png
[image6]: ./output_images/example_notcar_hog.png
[image7]: ./output_images/test1_cars.png
[image8]: ./output_images/test2_cars.png
[image9]: ./output_images/test3_cars.png
[image10]: ./output_images/test4_cars.png
[image11]: ./output_images/test5_cars.png
[image12]: ./output_images/test6_cars.png
[image13]: ./output_images/test1_windows.png
[image14]: ./output_images/test1_heat.png
[image15]: ./output_images/test2_windows.png
[image16]: ./output_images/test2_heat.png
[image17]: ./output_images/test3_windows.png
[image18]: ./output_images/test3_heat.png
[image19]: ./output_images/test4_windows.png
[image20]: ./output_images/test4_heat.png
[image21]: ./output_images/test5_windows.png
[image22]: ./output_images/test5_heat.png
[image23]: ./output_images/test6_windows.png
[image24]: ./output_images/test6_heat.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 20 through 136 of the file called `vehicle_detection.py`. The supporting functions are in a file called `vehicle_detection_functions.py`, specifcally, the get_hog_features function can be found on lines 31 through 49.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Example Car
![Example Car][image1]
Example Not-Car
![Example Not-Car][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example of a car and a not-car using the `YCrCb` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![Example Car for HOG][image3]
![Example HOG for Car][image4]
![Example Not-Car for HOG][image5]
![Example HOG for Not-Car][image6]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters including various color spaces, numbers of HOG orientations, HOG pixels per cell, HOG cells per block, and HOG channels.  I ended up with the ones above because of the combinations I tried, they seemed to provide the best results.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a combination of HOG features, spatial binning, and color histogram features on lines 139 through 202.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search at a few different scales since the cars should be different sizes in the image depending on how far away they are.  So, I searched a smaller area only near the horizon at the smallest scale and then increased the the max y-value for each increasing scale, searching the entire section of the raod part of the image for the largest scale.  The code for this is on lines 258 through 495.  It includes the `find_cars()` function which searches for a single scale and set of start and stop values for x and y.  It also includes the `detect_vehicles()` function which calls `find_cars()` multiple times for a few different scales and areas of the image as described above.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![Test 1 Cars][image7]
![Test 2 Cars][image8]
![Test 3 Cars][image9]
![Test 4 Cars][image10]
![Test 5 Cars][image11]
![Test 6 Cars][image12]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I also kept track of previous detections and included the last several frames worth of detections when calculating heat in order to boost and smooth positive identification of previously detected cars.

Here's an example result showing the heatmap from a series of images with their detections that the heatmap is based on.

### Here are six frames with window detections and their corresponding heatmaps:

![Test 1 Windows][image13]
![Test 1 Heatmap][image14]
![Test 2 Windows][image15]
![Test 2 Heatmap][image16]
![Test 3 Windows][image17]
![Test 3 Heatmap][image18]
![Test 4 Windows][image19]
![Test 4 Heatmap][image20]
![Test 5 Windows][image21]
![Test 5 Heatmap][image22]
![Test 6 Windows][image23]
![Test 6 Heatmap][image24]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I started out by using the code I had written and that was provided in the Lesson.  That at least gave me some results with some parameters that I already spent a little bit of time tweaking.  Doing that, I was able to pretty quickly get a working set of code that could at least run on the test images.  Being able to run on the test images gave me quick feedback about how the various combinations of parameters and features were performing. Once I got to testing on the full video, it got time consuming. So, I tried several combinations and settled with what I have now.

I was running into issues with lots of false positives.  One thing I did to try to combat that was to augment the data when extracting features for training with mirrored images.  That did seem to help a bit.  To help reduce false positives in the video I kept track of previous car detections and also the labeled heatmap spots and included the last several frames worth when calculating subsequent heatmaps.  In doing that, I was hoping to both amplify the true car detections in the heatmap and also to smooth out their labeling a bit.  This also meant I could increase the heatmap threshold to reduce the false positives.  That definitely improved the video, but I still have some false positives left in there.

My classifier had a pretty good test accuracy of around 99.4% So, I think it was trained well for the data available.  However, I think if the data was augmented more or possibly manually curated to create the training and test sets that might improve the accuracy on the video (and the test images).

 

