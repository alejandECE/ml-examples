## ml-examples
This is a extensive collection of examples demostrating core concepts, functionality and applications of machine learning, deep learning and data science in general. Most of these I used as code snippets in my lectures. The examples are grouped together in folders convering (in summary):

* [Bayesian Decision Theory](/bayesian%20decision%20theory) (Normal distributions, MVN, Bayes theorem, bayesian iterative learning ([Video](https://youtu.be/eVF82IoU3-Y)), probability of error, decision rules, regions and boundaries, GDA, LDA, regularized LDA, Naive Bayes, among others).

* [Linear Regression](/linear%20regression) (OLS, Robust Linear Regression, Laplace distribution, Huber Loss, Ridge Regression, LASSO, Bayesian Linear Regression).
This folder includes two examples of train and deployment of Google Cloud Platform (GCP) AI-Platform:
    * A regression example (using Tensorflow) found [here](/linear%20regression/keras%20gcp%20example%201).
    * A prediction example (using Tensorflow) found [here](/linear%20regression/keras%20gcp%20example%202).
    
* [Introduction to Gradient Descent](/gradient%20descend) (Numerical gradient, otimizers, line search, Newtwon's method, Adagrad, cost functions, among other). [Video 1](https://youtu.be/hnCuQcrs9kA), [Video 2](https://youtu.be/4W3Gf5-Z75o).

* [Logistic Regression](/logistic%20regression) (Logistic Regression (custom, using scipy.optimize toolbox and sklearn implementation), Softmax Regression, regularization, bias regularization effect). [Video 1](https://youtu.be/jGkTFk-MLh0), [Video 2](https://youtu.be/T1C6fTOUXkM).

....

   * [ConvNets](/conv%20nets). Some examples using ConvNets including:
      * Toy introductory examples to [convolution](conv%20nets/intro/image%20convolution.ipynb) and [pooling](conv%20nets/intro/image%20pooling.ipynb).
      * [Object localization](conv%20nets/object%20localization).
      * [Toy examples](conv%20nets/object%20detection/sliding%20windows) of object detection using [sliding windows](conv%20nets/object%20detection/sliding%20windows/sliding_window.py) and [convolutional sliding windows](/conv%20nets/object%20detection/sliding%20windows/convolutional_sliding_window.py). 
      * Object detection using a [complete-from-scratch implementation of Yolo V1 in Tensorflow](/conv%20nets/object%20detection/yolo/v1).
      * A pretty cool [oclussion visualization example](conv%20nets/visualizations/occlusion). The goal is to visualize what regions of the image are "more relevant" for the network when performing a classification task. The idea behind is to estimate something called the feature sensitivity of the posterior probabilities (FSPP), in other words how much the posterior probabilities (assuming softmax outputs) change when we "mess around" with some feature(s). This can be done in many ways, including some more matematically justified than others. For instance, in [this paper](https://ieeexplore.ieee.org/document/5282531) FSPP is used for feature selection and they proof that to determine the rank of a feature is enough to train a classifier and then permute a feature accross observations, evaluating the effect on the classifier output as we mess around with each feature (one at a time). In the occlussion experiment we simply "occlude" (set to zero!) some square region of the image and analyze the effect on the classifier output; if it changes a lot you could assume that region is important, otherwise not so much.
   
**Note**: All of these examples heavily use the tf.data module to improve the data ingestion pipeline. Also many of the datasets were conditioned (tfrecords generated as the end result) using ApacheBeam preprocessing pipelines capable or running locally and on Dataflow (GCP). Some of the TensorFlow Datasets weren't working properly of did not give me enough freedom to manipulate the data. Review [this](/records%20generators) for more details on the latter.*
   

The remaining description will be added shortly...
