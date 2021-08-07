# Real Time Age Classification Using Convolutional NeuralNetworks


Automatic age classification has become relevant to an increasing amount of applications, but the performance of existing methods on real-world images is still significantly lacking, especially when compared to the tremendous leaps in performance recently reported for the related task of face recognition. 
In this research project we are examining different deep-Convolutional Neural Networks and focusing on the performance of them. By examining the CNN architecture that was developed by Gil Levi and Tal Hassner, we gathered knowledge on their 3-layer fully connected convolutional neural network. We acknowledged that they are assuming their input images would contain faces, thus producing a somewhat not satisfying level of accuracy. 
By examing other's Convolutional Neural Networks and tuning Gil and Tal's model, we were able to construct our own model which could slightly outperform theirs. Furthermore, we were able to develop the model through a webcam, which could be used in real-time age classification. 

### Video Demo age detection (Coming Up)


### Report and detailed Introduction to this report can be accessed the following link.

Full Report: https://yushan1089.github.io/file/Age_Classification.pdf

Presentation: https://yushan1089.github.io/file/ageclassificationpresentation.pdf 

### Requirment
There are several things need to be installed to run the code of this project.

The TensorFlow module is required, due to the difference of installation between windows and macOS, I would simply refer a tutorial here: https://www.tensorflow.org/install

Also need OpenCV , keras, imutils package. Those could be done through pip install.

Webcam or other camera are preferred when using the code.

Use the below code to check if your PC or laptop has a GPU or not. If not, or memory is not sufficient, try to increase the batch size during training.
> import tensorflow as tf
>
> print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

### Instructions
This is a python project using Tensorflow and keras. If you want to use this project there are several ways to check out.

Directly using the trained model to predict age or do age classification task. 
Then go to your target directory, and use the following command:

> python detect_age_video.py

Wish to train your own age detector model then feel free to run the notebooks in the directory. 

The dataset we used in this project is the Adience Benchmark Project. You can find open access in the below link or search on Kaggle.

https://talhassner.github.io/home/projects/Adience/Adience-data.html

The performance difference between CNN design and VGG-16 model design for the model is discussed in the report above. Feel Free to check out.


### Models used:
* Gil and Tal's model
* AlexNet
* VGG-16 model

### Future Implementation: 
* Accuracy
* Image pre-processing (rather complicated procedure with center cropping)
* Perform Gaussian Filter to extract the outline of images
* Cost reduction
* Deployment to IOS and Android devices
* Age Classification with LBP

