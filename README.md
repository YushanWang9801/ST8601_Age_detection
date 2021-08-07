# Real Time Age Classification Using Convolutional NeuralNetworks


Automatic age classification has become relevant to an increasing amount of applications, but the performance of existing methods on real-world images is still significantly lacking, especially when compared to the tremendous leaps in performance recently reported for the related task of face recognition. 
In this research project we are examining different deep-Convolutional Neural Networks and focusing on the performance of them. By examining the CNN architecture that was developed by Gil Levi and Tal Hassner, we gathered knowledge on their 3-layer fully connected convolutional neural network. We acknowledged that they are assuming their input images would contain faces, thus producing a somewhat not satisfying level of accuracy. 
By examing other's Convolutional Neural Networks and tuning Gil and Tal's model, we were able to construct our own model which could slightly outperform theirs. Furthermore, we were able to develop the model through a webcam, which could be used in real-time age classification. 

Models used:
* Gil and Tal's model
* VGG-16 model

Future Implementation: 
* Accuracy
* Image pre-processing
* Data processing - other methods than center croping
* Perform Gaussian Filter to extract the outline of images
* Cost reduction
* Deployment to IOS and Android devices