---
title: "Prediction of Employee Attrition Using Deep Feed Forward Neural Network"
date: 2020-08-19
tags: [artificial neural network]
header:
  image: "images/StoneArchBridge.jpg"
excerpt: "Feed Forward Neural Network"
mathjax: true
toc: true
toc_label: "My Table of Contents"
toc_icon: "toc"
---

[Full Code](https://github.com/yoonkwon-yi/Project01-Prediction_of_Employee_Attrition_using_Artificial_Neural_Network/blob/master/Project01-Prediction_of_Employee_Attrition_using_Artificial_Neural_Network.ipynb)

## 1. Formulate Questions & Background

### What factors lead to attrition of an employee? Can we predict who is likely to leave?


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


<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/piechart.png" alt="piechart" align= "middle">

To see the correlations between the features and the target, a heatmap was created.

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/correlations.png" alt="heatmap of correlations of all features">

[Enlarge Diagram](https://yoonkwon-yi.github.io/images/Project01-HR/correlations.png)

As shown in the above heatmap, monthly income is strongly correlated (0.95) with Job Level. Also, Over time is positively correlated (0.25) with Attrition which is understandable.


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
The output of the above code block is shown below:

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/JobeRoleVSIncome.png" alt="JobRole VS Monthyl Income box plots">

As shown above, the average MonthlyIncome was the highest for Mangers and Research Directors, whereas Sales Representatives earned the lowest.

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

<br />
The below neural network architecture is a simplified version of the real model used in this project. The only correct layer would be the output as it indeed has only one node. Also, the number of layers are correctly illustrated as the model has 5 layers (1 input, 3 hidden, and 1 output layers). The number of nodes of the input layer should be 50 and the hidden layers should have 500 each.

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/ANNDiagram.JPG" alt="ANN architecture">


Since there are 50 columns in X_train (after OneHotEncoding and feature scaling), the input_shape will have 50 nodes indicated by the "input_shape" parameter.

The hidden layer uses rectified linear activation function (ReLu). The ReLu function is shown below. It outputs x if x is positive. Otherwise, it outputs zero.

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/relu.jpg" alt="relu diagram">

ReLu is beneficial for neural network with many neurons (as our model) since it has a sparse activation leading to a "lighter" network. By having few neurons to NOT activate, we would make the activation sparse and efficient. Other activation functions such as sigmoid and tanh would fire neurons in an analog way, leading to a dense activation, which is costly. In conclusion, ReLu was chosen as it is less computationally expensive compared to sigmoid or tanh.

The output layer has 1 node which would output a vector that is non-negative and sums to 1 (probability that the employee would leave the company). Since the current problem is a mutually exclusive binary problem (either leave or not leave the company), we can use the "sigmoid" function. Consequently, if the value of y_pred is above 0.5 (50%), we say that we predict the employee will leave the company and equal or lower than 0.5 would stay.

In regards to the optimizer, "Adam" was used as it achieve results more efficiently compared to other optimizers.

<br />

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/ANNLoss.png" alt="Model Loss Progress During Training">

Note how around Epoch number 25, Training Loss seem to plateau out reaching the minimum, indicating completion of neural network training. Also, even though we continue to train the model after Epoch number 25 up to 100, the graph indicates there are no overfitting.

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/ANNAccuracy.png" alt="Model Accuracy Progress During Training">

Similarly, the above graph of Model Accuracy reinforces that after Epoch number 25, the training of the model is complete.



## 6. Evaluate

Below is a confusion matrix resulted from the ANN model built above.
<br />
<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/cm.png" alt="ANN Confusion Matrix">

By using the information in the confusion matrix, precision, recall, and f1-score were calculated.


Precision is useful when costs of false positive (predicted true but in fact not true) are too high. For example, in terms of cancer tests, if the model's precision is so low, it would diagnoses healthy people as cancer patients. This would result in lots of mental stresses for the patients and spend in lots of resources.

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/Precision.png" alt="precision">


On the other hand, Recall is useful when false negatives' costs are too high. For example, in terms of diagnoses of a very deadly contagious virus. If the model predicts that a person with a virus is virus-free, the virus would spread too fast to contain in the future.

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/Recall.png" alt="recall">

The combination of Precision and Recall score is the F1 Score.

<img src="{{site.url}}{{site.baseurl}}/images/Project01-HR/F1.png" alt="F1">

As shown below, the weight average scores for the ANN model should roughly 0.85 for all precision, recall, and f1-score, indicating a fairly high performance from the model.

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
