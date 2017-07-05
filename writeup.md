# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./report_images/all_images.png "Visualization"
[image1]: ./report_images/dataset_chart.png "Visualization"
[image2]: ./report_images/grayscale.png "Grayscaling"
[image3]: ./report_images/increase.png "Increase"
<!-- [image3]: ./examples/random_noise.jpg "Random Noise" -->
[image4]: ./report_images/learning.png "Learning"
[image5]: ./report_images/trafficsigns.png "Traffic Sign"
[image6]: ./report_images/predict1.png "predict"
[image7]: ./report_images/predict2.png "predict"
[image8]: ./report_images/predict3.png "predict"
[image9]: ./report_images/predict4.png "predict"
[image10]: ./report_images/predict5.png "predict"
[image11]: ./report_images/predict6.png "predict"
[image12]: ./report_images/predict7.png "predict"
[image13]: ./report_images/predict8.png "predict"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### DataSet Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate unique classes/labels of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.   
There are 43 classes. Shown below is top images of each class.
![alt text][image0]

It is a bar chart showing how the data consists of.  
We can see that the number varies depending on the class.  

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to increase images and convert images by two ways.  
One is to rotate the images and two is to translate the images, because training set images are well centered and not rotated.  

rotate example  

translate example  

Below shows that training images are incresed 3 times.

![alt text][image3]

As a second step, I decided to convert images to grayscale and normalize because that are thought to be more robust.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution (5,5,1,6)     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution (5,5,6,16)	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| flatten	      	| outputs 400 				|
| Fully connected		| outputs 120        									|
| RELU					|												|
| Dropout					|	probability = 0.5											|
| Fully connected		| outputs 84        									|
| RELU					|												|
| Dropout					|	probability = 0.5											|
| Fully connected		| outputs 43        									|


### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following conditions.  
- optimize: AdamOptimizer    
- loss function: softmax_cross_entropy_with_logits  
- epochs: 50  
- batch size: 128  
- learning rate: 0.001  


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy = 0.994
* validation set accuracy = 0.964
* test set accuracy = 0.952

What I chose as the first architecture was LeNet which was the same as the lecture.
Validation set accuracy was a little lower than 0.93 because overfitting might happen.  
So I decided to add dropout layers which increased validation accuracy. Maybe thanks to prohibit overfitting and ensemble effect.  
Then I decided to increase training data set because images in some classes were a few numbers.
I also changed a learning rate, but under 0.001 made learning slow and above 0.001 made learning unstable.  

I thought epochs 30 and epochs 50 made almost no difference by plotting graph below, but tain accuracy and train loss were slightly well than epoch 30, so I dedided epochs as 50.  
![alt text][image4]  


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:  
(resized to 32x32 because my network handled that size.)  

![alt text][image5]

Stop, General caution, Speed limit(70km/h) might be difficult to predict because it was not clear, small, or perspectived.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Road work      		| Road work   									|
| Stop     			| End of no passing 										|
| General caution					| Keep right											|
| Speed limit(30km/h)	      		| Speed limit(80km/h)					 				|
| Roundabout mandatory			| Priority road      							|
| Turn right ahead			| Turn right ahead      							|
| Speed limit(70km/h)			| Ahead only      							|
| Right-of-way at the next intersection			| Right-of-way at the next intersection      							|


The model was able to correctly guess 3 of the 8 traffic signs, which gives an accuracy of 37.5%. This compares not favorably to the accuracy on the test set of 95.2%. This results are too smaller than expected. Maybe images from web is too difficult for my network.  
If I augmented images by such as scaling, perspecting, it would be got better result.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is definitely sure that this is a Road work and the image is Road work.
![alt text][image6]

For the second image, the model is relatively sure that this is a End of no passing, but the image is Stop.
That may be because when image were resized to 32x32, "STOP" picture made to be unclear.
![alt text][image7]

For the 3rd image, the model is relatively sure that this is a Keep right, but the image is General caution. That may be because this image is perspectived and triangle feature is relatively small.
![alt text][image8]

For the 4th image, the model is relatively sure that this is a Speed limit(80km/h), but the image is 30km/h. That may be because number 3 and number 8 are relatively similar.
![alt text][image9]

For the 5th image, the model is relatively sure that this is a Priority road, but the image is Roundabout mandatory. The probability of 5th is Roundabout mandatory, so if increasing the number of Roundabout mandatory test set, it might be predicted as Roundabout mandatory.
![alt text][image10]

For the 6th image, the model is relatively sure that this is a Turn right ahead, and the image is Turn right ahead.  
![alt text][image11]

For the 7th image, the model is relatively sure that this is a Ahead only, but the image is Speed limit(70km/h). That may be because this image is not square and sign is small, so when resized to 32x32 it is difficult to see as Speed limit(70km/h) even if by human.  
![alt text][image12]

For the 8th image, the model is definitely sure that this is a Right-of-way at the next intersection and the image is Right-of-way at the next intersection.  
![alt text][image13]
