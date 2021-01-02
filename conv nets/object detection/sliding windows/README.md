### Convolutional sliding windows
This (sub)repo has a demonstration of the use of convolutional sliding windows for building a dog detector (localizes dogs in a image). It is a toy example not aimed to improved the state of the art but instead to provide understanding in the subject matter.
I first implement a simple inefficient [sliding window approach](./sliding_window.py) and the use a fully convolutional network to evaluate many windows simultaneously.

A quick summary on some of the main script:
* Scripts [cnn1_training.py](./cnn1_training.py) and [cnn2_training.py](./cnn2_training.py) are just toy examples testing the classification performance on some know related datasets.
* Script [cnn3_training.py](./cnn3_training.py) is the training script to get a CNN with a classification and regression head (using fully connected layers). The resulting network is the one used in the sliding window script.
* Script [cnn4_training.py](./cnn4_training.py) trains the fully convolutional network.

All trainings and graphs are logged for visualization in Tensorboard. A video showing this can be found [here](https://youtu.be/Ec9BTzexaQY). Results applying the [convolutional sliding window approach](./convolutional_sliding_window.py) are shown [here](https://youtu.be/XHPVU3sZznE)
