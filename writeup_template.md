# **Traffic Sign Recognition** 

## Project summary

---

**Traffic Sign Recognition Project**

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

[image12]: ./examples/pic1.jpg "Traffic Sign 1"
[image13]: ./examples/pic2.jpg "Traffic Sign 2"
[image14]: ./examples/pic3.jpg "Traffic Sign 3"
[image15]: ./examples/pic4.jpg "Traffic Sign 4"
[image16]: ./examples/pic5.jpg "Traffic Sign 5"
[image17]: ./examples/pic6.jpg "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sguysc/Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

A bar chart showing how many images are in each class for the training-set. A map between class number and associated name of sign can be seen in the [CSV file](signnames.csv)

![alt text][image1]

also, here are three random classes to visualize 10 random images from that class (this will come in handy when deciding how to augment data):

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

The new histogram of the training data is the following ... 

![alt text][image11]

I then decided to convert the images to grayscale using a technique I saw online. The reason I did this is because that color should not make a difference for the classifier (a human can detect the signs using other features than color). 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image5]

Finally, I normalized the image data for numerical reasons. It usually helps the training be quicker to converge and to the correct minimum.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I started the project with the same LeNet model that we wrote in class. I tried data augmentation, normalization, decaying learning rate, different batch sizes, inserting dropouts in the model, however, no matter what I've done I could not top 92.5% in the test set. This led me to believe that the network is not "deep" enough to learn to do better. So I altered the network, especially the first 3 convolutional layers and made them much deeper (depth).
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 normalized grayscale image 			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x36 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x636 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x72	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x72    				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 3x3x144	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x144   				|
| Flatted		      	| outputs 2x2x144=576					   		|
| Fully connected		| outputs 84   									|
| RELU					|												|
| Dropout				| probability 0.5								|
| Fully connected		| outputs 43 classes   							|
| Softmax				|         										|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

As I mentioned, I tried many different approaches with the LeNet architecture. Finally, when switching to the new model I used this logic:

BATCH_SIZE = 64 - I read an article from Yann Lecun about not using batch size over 32 because in some situations it may not converge to the true minima. On the other hand, I wanted to speed things up a bit, so 64 was the compromise.

optimizer - AdamOptimizer because it is good enough for this purpose.

EPOCHS = 15 - I saw that there is no significant gain in training more epochs. It kind of reached a plateu after that.

learning rate = 0.001 - in the new network, this learning rate did the job. In the LeNet architecture I tried reducing the rate in higher accuracy range but it did not suffice to reach 93%.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.962 
* test set accuracy of 0.945

As described, I tried data augmentation, normalization, decaying learning rate, different batch sizes, inserting dropouts in the model all this in the LeNet model, however, no matter what I've done I could not top 92.5% in the test set. So I altered the network, especially the first 3 convolutional layers and made them much deeper (depth).

* What was the first architecture that was tried and why was it chosen? 

LeNet, because the problem seemed similar the the MNIST problem. That proved wrong (atleast for getting >93% accuracy)

* What were some problems with the initial architecture?

It seemed not deep enough, perhaps not having enough parameters to cope with the wider set of 43 classes and variety. Especially, even though I did recieve 95% in the validation set, the test set was much lower (<92.5%)

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

The model was altered a bit, by setting the 3 convolutional layers in the beginning much deeper. The rest was pretty much untouched. I did enter a dropout layer to avoid overfitting (I will discuss this later in the document). Once switching to the new model, validation set and test set were much closer and I immediatly recieved satisfactory results.

* Which parameters were tuned? How were they adjusted and why?

For the new model, a constant learning rate of 1e-3 was satisfactory. The parameters of the new model were copied from the LeNet architecture. The only thing is I squared the output layer (6->36) and also the subsequent convolutional layers.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

convolutional layers are very good with picture problems because they work on a small subset of the picture, learns simple things from it, and adds complexity as each layer is being added.
I used dropout as good engineering practice so I will not fit. I did not actually test if I could live without it. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web (after resizing them accordingly):

![alt text][image12] ![alt text][image13] ![alt text][image14] 
![alt text][image15] ![alt text][image16] ![alt text][image17] 

The 5th image might be difficult to classify because it's not in the classes, however, it is similar to class 39 so hopefully the data augmentation and rotation makes it possible to recognize it.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)	| Speed limit (20km/h)							| X
| Stop     				| Stop 											| V
| General caution		| General caution								| V
| Children crossing		| Children crossing				 				| V
| Keep left *			| No entry		     							| X
| Yield					| Yield			      							| V

Overall - 66.7% (and if not considering the 5th sign which is not in the class list, it is still 80%)

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is much lower than the validation set accuracy and also the test set accuracy. I suspect 2 reasons:
1. Statistics - I only checked 6 signs. Perhaps trying many more samples will correct the statistics.
2. Overfitting - Perhaps the model is overfitted. To backup this theory, I noticed that my previous model (LeNet) which had less accuracy in the test set, did 100% in my sign set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 3 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

true value - class #4: Speed limit (70km/h)

1st guess: 0 (100%)

2nd guess: 4 (0%)

3rd guess: 1 (0%)

true value - class #14: Stop

1st guess: 14 (97%)

2nd guess: 1 (1%)

3rd guess: 2 (1%)

true value - class #18: General caution

1st guess: 18 (100%)

2nd guess: 27 (0%)

3rd guess: 26 (0%)

true value - class #28: Children crossing

1st guess: 28 (100%)

2nd guess: 24 (0%)

3rd guess: 27 (0%)

true value - class #39: Keep left

1st guess: 17 (71%)

2nd guess: 38 (27%)

3rd guess: 9 (1%)

true value - class #13: Yield

1st guess: 13 (100%)

2nd guess: 3 (0%)

3rd guess: 12 (0%)


