## Frame the problem 🖼️

Let's start with a example dataset and analyze , extract all the insights that what that data is speaking to us. Hence we are going to get the dataset from **kaggle** and perform **exploratory data analysis**.

## About Data 🗄️

We are using **cars4u** dataset which you can access it from [here](https://www.kaggle.com/datasets/sukhmanibedi/cars4u). This dataset is in **xls** format. Dataset contains many features such as `Name`, `Location`, `Year`, `Kilometers_Driven`, `Fuel_Type`, `Transmission`....... and one **target** column is `price` of car.

## View the data 👁️‍🗨️

We can view the sample data using below code -

```python
data.head()
```
**Output**

![Alt text](assets/head.png)

## Info and describe

As we can see, data contains numerical features as well as categorical features, let's view it and also understand `max` , `min`, `mean` etc of all the features for statistical analysis.

```python
data.info()
```
**Output**

![Alt text](assets/info.png)

```python
data.describe().T
```

**Output**

![Alt text](assets/describe.png)

**Note** - Describe function analyzes the statistical insights of only numerical features.





