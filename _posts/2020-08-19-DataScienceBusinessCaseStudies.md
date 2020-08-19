---
title: "Data Science Project 01: Business Case Study Human Resource"
date: 2020-08-19
tags: [data science, logistic regression classifiers, ANN, random forest classifier]
header:
  image: "images/StoneArchBridge.jpg"
excerpt: "Data Science, HR, Case Study"
mathjax: true
---


# Question: Who will most likely to leave the company?

Data Source: [here](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)


## Visualization

```python
#read csv
employee_df = pd.read_csv('Human_Resources.csv')

#historgram of every features
employee_df.hist(bins = 30, figsize = (20,20), color = '#0077b6')
```

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/AllFeaturesGraphed.png" alt="histogram of all features">

[Enlarge Diagram](https://yoonkwon-yi.github.io/images/Project01-HR/AllFeaturesGraphed.png)

* Several features such as 'MonthlyIncome' and 'TotalWorkingYears' are skewed to the right.

* By looking at the graph we can see that we can drop features such as 'EmployeeCount','Standardhours','Over18', and'EmployeeNumber'

```python
employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)
```


<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/correlations.png" alt="heatmap of correlations of all features">

[Enlarge Diagram](https://yoonkwon-yi.github.io/images/Project01-HR/correlations.png)


<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/AgeAttrition.png" alt="Count plot of age and attrition">



```python
plt.figure(figsize=(15, 10))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
sns.boxplot(x = 'MonthlyIncome', y = 'JobRole', data = employee_df, palette='pastel')
sns.set(font_scale=2)
```

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/JobeRoleVSIncome.png" alt="JobRole VS Monthyl Income box plots">


## Train & Evaluate Artificial Neural Network
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
