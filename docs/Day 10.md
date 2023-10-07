## What is Linear Regression 💡 💡

[Referring](https://mlu-explain.github.io/linear-regression/)

**Linear Regression** is a simple and powerful model for predicting a numeric response from a set of one or more independent variables. This article will focus mostly on how the method is used in machine learning, so we won't cover common use cases like causal inference or experimental design. And although it may seem like linear regression is overlooked in modern machine learning's ever-increasing world of complex neural network architectures, the algorithm is still widely used across a large number of domains because it is effective, easy to interpret, and easy to extend. The key ideas in linear regression are recycled everywhere, so understanding the algorithm is a must-have for a strong foundation in machine learning.

### Let's Be More Specific

**Linear regression** is a supervised algorithm that learns to model a dependent variable, _**y**_, as a function of some independent variables (aka "features"), <b>x<sub>i</sub></b>​, by finding a line (or surface) that best "fits" the data. In general, we assume _**y**_ to be some number and each <b>x<sub>i</sub></b> can be basically anything. For example: predicting the price of a house using the number of rooms in that house (_**y**_ : price, <b>x<sub>1</sub></b> ​: number of rooms) or predicting weight from height and age (_**y**_ : weight, <b>x<sub>1</sub></b> : height, <b>x<sub>2</sub></b> : age).

In general, the equation for linear regression is

_**y**_ = <b>β<sub>0</sub></b> ​+ <b>β<sub>1</sub></b> <b>​x<sub>1</sub></b> ​ + <b>β<sub>2</sub></b><b>​x<sub>2</sub></b> ​+ ... + <b>β<sub>p</sub></b><b>​x<sub>p</sub></b>​ + </b>ϵ</b>

where:

- _**y**_: the dependent variable; the thing we are trying to predict.
- <b>x<sub>i</sub></b>: the independent variables: the features our model uses to model y.
- <b>β<sub>i</sub></b>​: the coefficients (aka "weights") of our regression model. These are the foundations of our model. They are what our model "learns" during optimization.[ℹ]
- _**ϵ**_: the irreducible error in our model. A term that collects together all the unmodeled parts of our data.


Fitting a linear regression model is all about finding the set of cofficients that best model _**y**_ as a function of our features. We may never know the true parameters for our model, but we can estimate them (more on this later). Once we've estimated these coefficients, <b>β<sub>i</sub><sup>^</sup></b> ​, we predict future values, <b>y<sup>^</sup></b>, as:

 <b>y<sup>^</sup></b> ​= <b>β<sub>0</sub><sup>^</sup></b> ​​+ <b>β<sub>1</sub><sup>^</sup>x<sub>1</sub></b> ​+ <b>β<sub>2</sub><sup>^</sup>x<sub>2</sub></b> ​+ ..... + <b>β<sub>p</sub><sup>^</sup>x<sup>p</sup></b>

 So predicting future values (often called inference), is as simple as plugging the values of our features xixi​ into our equation!