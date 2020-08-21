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

## 3. Explore & Visualize


As shown in the below heatmap, there is a strong positive correlation between 'purchases_frequency' and 'purchases_installments_frequency' as well as 'one-off purchases' and  'purchases'

<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/03Correlation.jpg" alt="heatmap correlation">
[Enlarge Diagram](https://yoonkwon-yi.github.io/images/Project02-Marketing/03Correlation.jpg)

## 4. Train Algorithm

### K-means Clustering

The "K" in the name "K-menas Clustering" stands for the number of clusters that categorizes the dataset. One way of determining the value "K" is using the 'Elbow Method'.

The "Elbow" is graphed by iterating K-means clustering by varying K values. Then the 'distortion' of each clusters are plotted against the K values. This result in a graph similar to the one below.

We pic the value K on x-axis as the optimal K when the rate of increase in the slope of the graph slows down significantly. If the graph is imagined as a bent elbow, the joint of the elbow would be the optimal K value. In our project, 5 was chosen as the K.

<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/04Elbow.png" alt="finding k using elbow">

### Principal Component Analysis (PCA)
Now before visualizing the clusters, PCA must be performed on the dataset to reduce the 17th dimension data (since we have 17 columns) to 2nd dimension.


```python
# Obtain the principal components
pca = PCA(n_components=2)
principal_comp = pca.fit_transform(creditcard_df_scaled)

# Create a dataframe with the two components
pca_df = pd.DataFrame(data = principal_comp, columns =['pca1','pca2'])

# Concatenate the clusters labels to the dataframe
pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)
```
Once we perform the dimensionality reduction, we can plot it as a 2-D graph as shown below.

<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/06pca.png" alt="post pca scatterplot">
[Enlarge Diagram](https://yoonkwon-yi.github.io/images/Project02-Marketing/06pca.png)

### Auto Encoder

Above, we were able to cluster the customers of the bank into 5 groups. What if we want to cluster them in a even smaller number of groups? We could use Auto Encoder to reduce the dimensionality, similar to PCA, and cluster them.

Below image shows a simplified diagram of an Auto Encoder.  

<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/autoencoder.png" alt="auto encoder">

It is comprised of encoder, bottleneck, and decoder. As we move from the input layer to the bottleneck, as the name implies, the dimensionality is reduced (compressed). The decoder is used for training, and what goes into the encoder should be the same when it comes out the decoder. In the end, what we get at the bottleneck is what is used (reduced dimension data).

```python

#initially we have 17 columns for our data
input_df = Input(shape=(17,))

# Glorot normal initializer (Xavier normal initializer) draws samples from a truncated normal distribution
x = Dense(7, activation='relu')(input_df)
x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(2000, activation='relu', kernel_initializer = 'glorot_uniform')(x)

encoded = Dense(10, activation='relu', kernel_initializer = 'glorot_uniform')(x)

x = Dense(2000, activation='relu', kernel_initializer = 'glorot_uniform')(encoded)
x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)

decoded = Dense(17, kernel_initializer = 'glorot_uniform')(x)

# autoencoder
autoencoder = Model(input_df, decoded)
```

The variable "autoencoder" is the whole neural network shown in the above diagram. However, we want the "bottleneck" layer as our data would be the most compressed.


```python
#encoder - used for our dimention reduction
encoder = Model(input_df, encoded)

autoencoder.compile(optimizer= 'adam', loss='mean_squared_error')

autoencoder.fit(creditcard_df_scaled, creditcard_df_scaled, batch_size = 128, epochs = 25,  verbose = 1)

#get the compressed data called pred
pred = encoder.predict(creditcard_df_scaled)
```
After reducing the dimension using the Auto Encoder, the Elbow method was used to find the optimal K value for the new reduced data. The elbow graph is plotted below.

<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/06.5elbow.png" alt="elbow after autoencoder">

Then the new elbow graph (post-Autoencoder) was plotted on the same figure with the pre-Autoencoder. As shown below, the "elbow" on the graph is reached earlier for the post-Autoencoder (pink) than the pre-Autoencoder (blue).

<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/06.7bothelbow.png" alt="2 elbows on one graph">

Thus the K value was determined to be 3, and the K-means clustering was performed. Again, PCA was used to visualize the final clustering outcome.

<img src="{{site.url}}{{site.baseurl}}/images/Project02-Marketing/07pcaAutoencoder.png" alt="post pca after autoencoder">
[Enlarge Diagram](https://yoonkwon-yi.github.io/images/Project02-Marketing/07pcaAutoencoder.png)



## 5. Conclusion

In this project, I utilized Autoencoder and K-means clustering technique to group bank customers to allow targeted marketing. PCA method was used to visualize high dimensional data into 2D space. 









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
