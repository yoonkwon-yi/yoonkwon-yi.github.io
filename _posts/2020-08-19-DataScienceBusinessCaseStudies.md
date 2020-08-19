---
title: "Data Science Project 01: Business Case Study Human Resource"
date: 2020-08-19
tags: [data science, logistic regression classifiers, ANN, random forest classifier]
header:
  image: "images/StoneArchBridge.jpg"
excerpt: "Data Science, HR, Case Study"
mathjax: true
---


# Question: Predict Which Employees Will Leave

Data Source: [here](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)


## Visualization

```python

#read csv
employee_df = pd.read_csv('Human_Resources.csv')

#historgram of every features
employee_df.hist(bins = 30, figsize = (20,20), color = '#0077b6')

```

* Several features such as 'MonthlyIncome' and 'TotalWorkingYears' are skewed to the right.

* By looking at the graph we can see that we can drop features such as 'EmployeeCount','Standardhours','Over18', and'EmployeeNumber'

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/AllFeaturesGraphed.png" alt="histogram of all features">


```python
employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)


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


Python code block:

```python
import numpy as np

def test_function(x,y):
  z= np.sum(x,y)
  return z
```

<!--
Here's some inline code 'x+y'

Here's an image:
<img src="{{site.url}}{{site.baseurl}}/images/DominicYiPortrait.jpg" alt="linearly separable data">


Here's another image using Kramdown:
![alt]({{site.url}}{{site.baseurl}}/images/DominicYiPortrait.jpg)

Here's some math:
$$z=x+y$$

You can also put it inline $$z=x+y$$ -->
