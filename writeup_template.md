#**Traffic Sign Recognition** 

**Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./observation.png "Histogram of classes and number of observation"
[image2]: ./gray.png "Grayscaling"
[image3]: ./predict.png "Predict"
[image4]: ./1.jpg "Traffic Sign 1"
[image5]: ./2.jpg "Traffic Sign 2"
[image6]: ./3.jpg "Traffic Sign 3"
[image7]: ./4.jpg "Traffic Sign 4"
[image8]: ./5.jpg "Traffic Sign 5"


You're reading it! and here is a link to my [project code](https://github.com/bhatabhijithn/GermanTrafficSignsClassification/blob/master/Traffic_Sign_Classifier-Augmentation.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![Histogram of classes and number of observation][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I experimented with different preprocessing technique. Opencv knowledge is very important and i had to learn from scratch on how to use it for understand the nitty gritties of it. 
1. First convert the images to grayscale cv2.COLOR_BGR2GRAY
2. Then use cv2.equalizeHist(img) for making sure the image brightness and contrast is taken care
3. Then Normalise it
4. Add bias of .02 which improved prediction accuracy.

Here is an example of a traffic sign image before and after grayscaling.

I decided to generate additional data because as i trained existing data, it was peaking at 81%. Then started looking at adding more data. I used simple rotate image to fill in data. The minimum pictures for a set is 809 is generated

To add more data to the the data set, I used the following techniques as This will help in not overfitting the samples which have higher samples

```python
Angles used to rotate the images
    angles = [-10, 10, -5, 5, -15, 15, -20, 20]

    #iterate through each class
    for i in range(len(pics_in_class)):
        #Check if less data than the mean
        if pics_in_class[i] < int_mean:
            #Count how much additional data you want
            new_wanted = int_mean - pics_in_class[i]
            picture = np.where(y_train==i)
            more_X = []
            more_y = []

            for num in range(new_wanted):
                more_X.append(ndimage.rotate(X_train[picture][random.randint(0,pics_in_class[i] -1)], random.choice(angles), reshape=False))
                more_y.append(i)

            #Append the pictures generated for each class back to original shape
            X_train = np.append(X_train, np.array(more_X), axis=0)
            y_train = np.append(y_train, np.array(more_y), axis =0)
```        

Here is an example of an original image and an augmented image:

![alt text][image2]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution1 3x3     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				    |
| Convolution2 3x3	    | 1x1 stride, Valid padding, outputs 10x10x16 	|      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16				    |
| Fully connected1		| 400. Dropout 									|
| Fully connected2		| 100. Dropout 									|
| Fully connected3		| 43         									|
| logits				| 43        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used LeNet architecture, I added dropout  in the layers to improve accuracy and reduce over fitting.

To train the model, I used an Adam optimiser which is almost now a defacto optimiser for classification, with objective to reduce cross entropy. 
I used learning rate of 0.05 (0.02-0.005 tested, due to shuffling of training data, 0.04-0.06 seems to be good learning rate but it is difficult to get which learning rate gives accuracy more than 0.925. I have got 0.94 for same setting.)
I used 30 epochs with batch size of 150.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 94.6%
* test set accuracy of 92.6%

 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| General Caution		| General Caution								|
| Road work				| Road work 									|
| 50 km/h	      		| 80 km/h			     		 				|
| ROW at intersection	| ROW at intersection 							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The 50 km/h had a bent which makes bottom part of 5 as 8.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][image2]



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The conv1 layer makes profound impact. Here we can see why 50kmph was seen as 80kmph. Due to bend, its feature map show mix of 80/50 representation. Feature 2 maps too show the noise to be high in 50 kmph sign.
