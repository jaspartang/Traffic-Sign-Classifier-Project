# **Traffic Sign Recognition** 

## Project summary

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

**Source code (Notebook)**
* [project code](https://github.com/sguysc/Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/keepright_train.jpg "keep right samples"
[image3]: ./examples/slipperyroad_train.jpg "slippery road samples"
[image4]: ./examples/speedlimit20_train.jpg "speed limit 20 samples"
[image5]: ./examples/grayscale.jpg "Grayscaling"
[image6]: ./examples/org_image.png "original random image"
[image7]: ./examples/bight_image.png "brightness"
[image8]: ./examples/rotate_image.png "rotate"
[image9]: ./examples/translate_image.png "translate"
[image10]: ./examples/all_image.png "all filters"
[image11]: ./examples/newHistogram.png "new histogram"

[image6]: ./examples/random_noise.jpg "Random Noise"
[image7]: ./examples/placeholder.png "Traffic Sign 1"
[image8]: ./examples/placeholder.png "Traffic Sign 2"
[image9]: ./examples/placeholder.png "Traffic Sign 3"
[image10]: ./examples/placeholder.png "Traffic Sign 4"
[image11]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sguysc/Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

A bar chart showing how many images are in each class for the training-set. A map between class number and associated name of sign can be seen in the [CSV file](signnames.csv)

![alt text][image1]

also, here is a three random to visualize random 10 images from that class (this will come handy when deciding how to augment data:
![alt text][image2]
![alt text][image3]
![alt text][image4]

### Design and Test a Model Architecture

#### 1. Preprocessing the image data

As a first step, I decided to generate additional data because looking and the histogram above, I thought the imbalance in number of images between classes might lead the network to prefer those with high numbers. I wanted to generate new fake data out of the classes that have less than the mean number of images per class (~810 images). This way, the network won't overtrain on the small samples classes.

To add more data to the the data set, I used the following techniques: For each class that has less than ~810 images, and for each original image, add several random images based on that so that
the total number of images will come to at least 800. I used several preprocessing funcions I found online using opencv library. The sequence is so:
1. equalize the histogram for each channel (RGB)
2. random scaling
3. random translate
4. random rotation
5. random brightness
I tried doing this to classes with number of images more the the mean but that got me no improvement in accuracy (perhaps that data is "rich" enough) so I dropped it.

Here is an example of an original image and an augmented image:
![alt text][image6]
applying random brightness
![alt text][image7]
applying random rotation
![alt text][image8]
applying random translation
![alt text][image9]
applying all filters at random 
![alt text][image10]

The difference between the original data set and the augmented data set is the following ... 
![alt text][image11]

I then decided to convert the images to grayscale using a technique I saw online. The reason I did this is because that color should not make a difference for the classifier (a human can detect the signs using other features than color). 

Here is an example of a traffic sign image before and after grayscaling.
![alt text][image5]

Finally, I normalized the image data for numerical reasons. It usually helps the training be quicker to converge and to the correct minimum.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


