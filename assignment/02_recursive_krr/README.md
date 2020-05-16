# "Recursive" Kernel Ridge Regression

## Assignment Idea

* Learn a simple 1D function
* Dataset of 10 points, 9 train, 1 test
* x-value is the representation
* Train 10 models and try to use the prediction of each model to somehow include
it in either the training set or representation itself
* Think of a way to learn a function instead of feature-label pair


## My Idea
* Can be formulated as a semi-supervised learning task
* This should minimize the predictive variance of points in between
data points