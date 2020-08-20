---
title: "Data Science Project 01: Business Case Study Human Resource"
date: 2020-08-19
tags: [data science, logistic regression classifiers, ANN, random forest classifier]
header:
  image: "images/StoneArchBridge.jpg"
excerpt: "Data Science, HR, Case Study"
mathjax: true
---



## 1. Formulate Questions

### What factors lead to 1. employees performing well or 2. leaving the company?

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/HRPic.jpg" alt="HR Stock Picture">

## 2. Gather Data

Data Source: From Kaggle (Click [here](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset))
-  This is a fictional data set created by IBM data scientists.

## 3. Clean Data

```python
employee_df.info()
```
Output of the above code:

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/dataInfo.JPG" alt="df.info output">

Row 1 showed "Attrition" which would be out Target(dependent variable) and all other rows would be Features (independent variables).

Columns containing dichotomous variables such as "Attrition", "Over18", and "OverTime" were converted to numerical by using the below code:


```python
employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['Over18'] = employee_df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)
```

To check if there are any missing data, below code was used.
```python
sns.heatmap(employee_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
```
<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/isNull.png" alt="check Null data">

As shown above, there weren't any missing data (if there was, it would have been blue in the chart).

## 4. Explore & Visualize

Histograms for all features as well as the target (Attrition) was plotted.



```python
employee_df.hist(bins = 30, figsize = (20,20), color = '#0077b6')
```

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/AllFeaturesGraphed.png" alt="histogram of all features">

[Enlarge Diagram](https://yoonkwon-yi.github.io/images/Project01-HR/AllFeaturesGraphed.png)

* Several features such as 'MonthlyIncome' and 'TotalWorkingYears' are skewed to the right.

* By looking at the graph we can see that we can drop features such as 'EmployeeCount','Standardhours','Over18', and'EmployeeNumber'


```python
employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)
```

Then the dataframe was divided into two groups: one that ended up leaving the company and another that stayed.


```python
left_df        = employee_df[employee_df['Attrition'] == 1]
stayed_df      = employee_df[employee_df['Attrition'] == 0]
```

A pie chart illustrating the composition of left and stayed was plotted.


<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/pichart.png" alt="piechart">

To see the correlations between the features and the target, a heatmap was created.

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/correlations.png" alt="heatmap of correlations of all features">

[Enlarge Diagram](https://yoonkwon-yi.github.io/images/Project01-HR/correlations.png)

Some interesting points from the above heatmap includes:

- Monthly income is strongly correlated (0.95) with Job Level.
- Over time is positively correlated (0.25) with Attrition which is understandable.


Another count plot was created to see the relationship between Age and Attrition.


<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/AgeAttrition.png" alt="Count plot of age and attrition">

As shown above. the younger the employees are, more percentage of attrition occured.


Then the MonthylIncome was visualized categorized by JobRole.

```python
plt.figure(figsize=(15, 10))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
sns.boxplot(x = 'MonthlyIncome', y = 'JobRole', data = employee_df, palette='pastel')
sns.set(font_scale=2)
```

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/JobeRoleVSIncome.png" alt="JobRole VS Monthyl Income box plots">

- The average MonthlyIncome was the highest for Mangers and Research Directors, whereas Sales Representatives earned the lowest.

## 5. Train Algorithm

Before training the algorithm, categorical data was converted into numerical by using One Hot Encoding.

'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', and 'MaritalStatus' were categorical.

```python
X_cat = employee_df[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
X_cat = pd.DataFrame(X_cat)
```

Then the outcome of OneHotEncoder was concatenated with the numerical features.

### Artificial Neural Network (ANN)

```python
import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=500, activation='relu', input_shape=(50, )))
model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])

epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

```

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/ANNLoss.png" alt="Model Loss Progress During Training">

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/ANNAccuracy.png" alt="Model Accuracy Progress During Training">



## 6. Evaluate
<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/ANNCM.png" alt="ANN Confusion Matrix">

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/ANNScores.JPG" alt="ANN classification report">






<!--
Here's some basic text.

And here's some *italics*

Here's some **bold** text. -->
<!--
What about a [link](https://github.com/yoonkwon-yi)?
Here's a bulleted list:

* First item
+ Second item
- Third item


Here's a numbered list:
1. First
2. Second
3. Third -->

<!--
Python code block:

```python
import numpy as np

def test_function(x,y):
  z= np.sum(x,y)
  return z
``` -->

<!--
Here's some inline code 'x+y'

Here's an image:
<img src="{{site.url}}{{site.baseurl}}/images/DominicYiPortrait.jpg" alt="linearly separable data">


Here's another image using Kramdown:
![alt]({{site.url}}{{site.baseurl}}/images/DominicYiPortrait.jpg)

Here's some math:
$$z=x+y$$

You can also put it inline $$z=x+y$$ -->
