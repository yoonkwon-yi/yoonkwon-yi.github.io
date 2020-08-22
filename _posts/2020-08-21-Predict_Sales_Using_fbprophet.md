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

StoreType - differentiates between 4 different store models: a, b, c, d



## 4. Train Algorithm


```python
from fbprophet import Prophet

def sales_prediction(Store_ID, sales_df, periods):
  # Function that takes in the data frame, storeID, and number of future period forecast
  # The function then generates date/sales columns in Prophet format
  # The function then makes time series predictions

  sales_df = sales_df[ sales_df['Store'] == Store_ID ]
  sales_df = sales_df[['Date', 'Sales']].rename(columns = {'Date': 'ds', 'Sales':'y'})
  sales_df = sales_df.sort_values('ds')

  model    = Prophet()
  model.fit(sales_df)
  future   = model.make_future_dataframe(periods=periods)
  forecast = model.predict(future)
  figure   = model.plot(forecast, xlabel='Date', ylabel='Sales')
  figure2  = model.plot_components(forecast)

```




<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/prediction.png" alt="sales prediction without holidays">

<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/trend1.png" alt="trend prediction without holidays">

<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/day week 1.png" alt="week prediction without holidays">

<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/day year 1.png" alt="year prediction without holidays">




#### Taking holidays into account
```python
# Get all the dates pertaining to school holidays
school_holidays = sales_train_all_df[sales_train_all_df['SchoolHoliday'] == 1].loc[:, 'Date'].values

# Get all the dates pertaining to state holidays
state_holidays = sales_train_all_df [ (sales_train_all_df['StateHoliday'] == 'a') | (sales_train_all_df['StateHoliday'] == 'b') | (sales_train_all_df['StateHoliday'] == 'c')  ].loc[:, 'Date'].values

state_holidays = pd.DataFrame({'ds': pd.to_datetime(state_holidays),
                               'holiday': 'state_holiday'})
school_holidays = pd.DataFrame({'ds': pd.to_datetime(school_holidays),
                               'holiday': 'school_holiday'})

 # concatenate both school and state holidays
 school_state_holidays = pd.concat((state_holidays, school_holidays))
```




```python
from fbprophet import Prophet

def sales_prediction(Store_ID, sales_df, holidays, periods):
  # Function that takes in the storeID and returns two date/sales columns in Prophet format
  # Format data to fit prophet

  sales_df = sales_df[ sales_df['Store'] == Store_ID ]
  sales_df = sales_df[['Date', 'Sales']].rename(columns = {'Date': 'ds', 'Sales':'y'})
  sales_df = sales_df.sort_values('ds')

  model    = Prophet(holidays = holidays)
  model.fit(sales_df)
  future   = model.make_future_dataframe(periods = periods)
  forecast = model.predict(future)
  figure   = model.plot(forecast, xlabel='Date', ylabel='Sales')
  figure2  = model.plot_components(forecast)
```


<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/2predict.png" alt="sales prediction WITH holidays">

<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/2trend.png" alt="trend prediction WITH holidays">

<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/2holiday.png" alt="holiday prediction WITH holidays">

<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/2weekly.png" alt="week prediction WITH holidays">
<img src="{{site.url}}{{site.baseurl}}/images/Project03-Sales/2yearly.png" alt="yearly prediction WITH holidays">





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
