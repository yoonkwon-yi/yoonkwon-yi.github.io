---
title: "Customer Segmentation for Marketing using K-means Clustering and Autoencoder"
date: 2020-08-20
tags: [clustering, dimensionality reduction]
header:
  image: "images/StoneArchBridge.jpg"
excerpt: "Unsupervised Machine Learning, Dimensionality Reduction"
mathjax: true
---

<!-- [Full Code](https://github.com/yoonkwon-yi/Project01-Prediction_of_Employee_Attrition_using_Artificial_Neural_Network/blob/master/Project01-Prediction_of_Employee_Attrition_using_Artificial_Neural_Network.ipynb) -->

## 1. Formulate Questions & Background

### How can we understand the customers better to launch a targeted marketing campaign?


<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/target.png" alt="target clusters">

## 2. Gather Data

Data Source: From Kaggle (Click [here](https://www.kaggle.com/arjunbhasin2013/ccdata))

- "The sample Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. The file is at a customer level with 18 behavioral variables."

## 3. Clean Data

<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/01Missing.png" style="max-width:100%;" alt="check missing data">


<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/02NotMissing.png" alt="check missing data again">

## 4. Explore & Visualize



<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/03Correlation.jpg" alt="heatmap correlation">
[Enlarge Diagram](https://yoonkwon-yi.github.io/images/Project02-Marketing/03Correlation.jpg)


<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/04Elbow.png" alt="finding k using elbow">







<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/06pca.png" alt="post pca scatterplot">
[Enlarge Diagram](https://yoonkwon-yi.github.io/images/Project02-Marketing/06pca.png)


<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/06.5elbow.png" alt="elbow after autoencoder">

<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/06.7bothelbow.png" alt="2 elbows on one graph">


<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/07pcaAutoencoder.png" alt="post pca after autoencoder">
[Enlarge Diagram](https://yoonkwon-yi.github.io/images/Project02-Marketing/07pcaAutoencoder.png)



## 5. Train Algorithm


## 6. Evaluate






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
