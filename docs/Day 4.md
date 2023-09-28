## Frame the problem 🖼️

Let's start with a example dataset and analyze , extract all the insights that what that data is speaking to us. Hence we are going to get the dataset from **kaggle** and perform **exploratory data analysis**.

## About Data 🗄️

We are using **cars4u** dataset which you can access it from [here](https://www.kaggle.com/datasets/sukhmanibedi/cars4u). This dataset is in **xls** format. Dataset contains many features such as `Name`, `Location`, `Year`, `Kilometers_Driven`, `Fuel_Type`, `Transmission`....... and one **target** column is `price` of car.

## Import the essentials

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')
```

## View the data 👁️‍🗨️

We can view the sample data using below code -

```python
data.head()
```
**Output**

![Alt text](assets/head.png)

## Info and describe 🤡

As we can see, data contains numerical features as well as categorical features, let's view it and also understand `max` , `min`, `count`, `mean` etc of all the features for statistical analysis.

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

From the statistics summary, we can infer the below findings :

- Years range from 1996- 2019 and has a high in a range which shows used cars contain both latest models and old model cars.
- On average of Kilometers-driven in Used cars are ~58k KM. The range shows a huge difference between min and max as max values show 650000 KM shows the evidence of an outlier. This record can be removed.
- Min value of Mileage shows 0 cars won’t be sold with 0 mileage. This sounds like a data entry issue.
- It looks like Engine and Power have outliers, and the data is right-skewed.
- The average number of seats in a car is 5. car seat is an important feature in price contribution.
- The max price of a used car is 160k which is quite weird, such a high price for used cars. There may be an outlier or data entry issue.


## Number of Unique values 

Why we need to find unique values ? 
- because when length of data is not equal to total unique value, it means data contains duplicate values and duplicate values is the senseless thing in the data.

```python
data.nunique()
```
**Output**

![Alt text](assets/nunique.png)

## Missing values calculation 🖩

Our machine learning model doesn't work wells when there is missing values in the data. Sometimes it also gives errors that data has `NaN` values. So to count number of missing values. we can use 

```python
data.isnull().sum() # it will display number of missing values in each feature
```

**Output**

![Alt text](assets/missing.png)

We can see that there are missing values in feature `engine`, `power`, `seats`, `new price` and `price`. In further days, we'll deal with those missing values that how we can remove them or replace them.

## Remove Unwanted columns 📎

There are many columns which we don't want to use for training our machine learning model, like `S.No` because this column don't have a meaning it is just numbering. So we'll remove them 

```python
data = data.drop(['S.No.'],axis=1) # axis = 1 is for column wise
```

**After removing column, dataset will look like**

![Alt text](assets/afterremoving.png)

## Feature Engineering

Feature engineering refers to the process of using domain knowledge to select and transform the most relevant variables from raw data when creating a predictive model using machine learning or statistical modeling. The main goal of Feature engineering is to create meaningful data from raw data. 

### Creating features

It would be difficult to find the car’s age if it is in year format as the Age of the car is a contributing factor to Car Price. 

Let's introduce a new column `car_age` for calculating age of car.

```python
from datetime import date
date.today().year
data['Car_Age']=date.today().year-data['Year']
data.head()
```

**Output**

![Alt text](assets/engineering.png)

Since car names will not be great predictors of the price in our current data. But we can process this column to extract important information using brand and Model names. Let’s split the name and introduce new variables **Brand** and **Model**

```python
data['Brand'] = data.Name.str.split().str.get(0)
data['Model'] = data.Name.str.split().str.get(1) + data.Name.str.split().str.get(2)
data[['Name','Brand','Model']]
```

**After splitting the `brand`, `name` and `model`, columns will look like**

![Alt text](assets/brand_model.png)

## Let's clean the data 🪥

Some names of the variables are not relevant and not easy to understand. Some data may have data entry errors, and some variables may need data type conversion. We need to fix this issue in the data.

`In the example`, The brand name ‘Isuzu’ ‘ISUZU’ and ‘Mini’ and ‘Land’ looks incorrect. This needs to be corrected

As you can see below 👇

![Alt text](assets/isuzu.png)

So 

```python
searchfor = ['Isuzu' ,'ISUZU','Mini','Land']
data[data.Brand.str.contains('|'.join(searchfor))].head(5) ## run this cell

# hence we'll replace the word and will correct it

data["Brand"].replace({"ISUZU": "Isuzu", "Mini": "Mini Cooper","Land":"Land Rover"}, inplace=True)
```

## Separate something

Before moving further to **univariate** and *bivariate* analysis, let's separate **numerical** and **categorical** features.

```python
cat_cols=data.select_dtypes(include=['object']).columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()

```

## Univariate Analysis

Now **Univariate** analysis means analyzing every feature of the dataset so that we'll get clarity regarding preprocessing things for each features. We'll create some distribution charts like

- **Skew value** to understand that is our distribution of numerical feature is normal (bell curve) or not (biased left or right).
- **Histogram** is a graph plot to define the distribution of each feature (we can also use kernle plot).
- **Box plot** to understand that feature contains outliers or not.

```python
for col in num_cols:
    print(col)
    print('Skew :', round(data[col].skew(), 2))
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 2, 1)
    data[col].hist(grid=False)
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[col])
    plt.show()
```

**Output**

![Alt text](assets/v.png)

![Alt text](assets/v-1.png)

![Alt text](assets/v-2.png)

![Alt text](assets/v-3.png)

![Alt text](assets/v-4.png)






