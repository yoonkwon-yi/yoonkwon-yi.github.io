---
title: "Sales Prediction Using Facebook Prophet"
date: 2020-08-21
tags: [Forecasting]
header:
  image: "images/StoneArchBridge.jpg"
excerpt: "forecasting time series, additive model"
mathjax: true
---

<!-- [Full Code](https://github.com/yoonkwon-yi/Project01-Prediction_of_Employee_Attrition_using_Artificial_Neural_Network/blob/master/Project01-Prediction_of_Employee_Attrition_using_Artificial_Neural_Network.ipynb) -->

## 1. Formulate Questions & Background

### How can we understand the customers better to launch a targeted marketing campaign?


## 2. Gather Data

Data Source: From Kaggle (Click [here](https://www.kaggle.com/c/rossmann-store-sales/data))

- "You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. Note that some stores in the dataset were temporarily closed for refurbishment."

## 3. Explore & Visualize

<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/correlations.png" alt="post correlations between columns">


<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/typeStores.png" alt="type stores ">



<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/prediction.png" alt="sales prediction without holidays">

<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/trend1.png" alt="trend prediction without holidays">

<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/day week 1.png" alt="week prediction without holidays">

<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/day year 1.png" alt="year prediction without holidays">


<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/2predict.png" alt="sales prediction WITH holidays">

<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/2.trend.png" alt="trend prediction WITH holidays">

<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/2holiday.png" alt="holiday prediction WITH holidays">

<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/2weekly.png" alt="week prediction WITH holidays">
<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/2yearly.png" alt="yearly prediction WITH holidays">

## 4. Train Algorithm



## 5. Conclusion







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
