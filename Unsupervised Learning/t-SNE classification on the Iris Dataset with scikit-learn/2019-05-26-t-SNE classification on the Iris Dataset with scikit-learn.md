---
layout: single
classes: wide
title: "t-SNE Classification on the Iris Dataset with scikit-learn"
header:
  teaser: /assets/images/5x3/dimension.png
  overlay_color: "#5e616c"
  overlay_image: /assets/images/1920x1080/dimension.png
  overlay_filter: 0.6
  actions:
  - label: "<i class='fas fa-arrow-alt-circle-right'></i> View in Jupyter"
    url: "https://nbviewer.jupyter.org/github/chrisman1015/Unsupervised-Learning/blob/master/t-SNE%20on%20the%20Iris%20Dataset%20with%20scikit-learn/t-SNE%20classification%20on%20the%20Iris%20Dataset%20with%20scikit-learn.ipynb"
excerpt: >
  By: Christopher Hauman<br />
categories:
  - Python
  - Unsupervised Machine Learning
tags:
  - Python
  - Machine Learning
  - Unsupervised
  - t-SNE
  - Classification
---

This will quickly run through using scikit-learn to perform t-SNE on the [Iris](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) dataset. This is an adapted example from Datacamp's course on [Unsupervised Learning in Python](https://www.datacamp.com/courses/unsupervised-learning-in-python). If you're not familiar with unsupervised learning, you should start [here](https://chrisman1015.github.io//python/unsupervised%20machine%20learning/K-Means-Classification-on-the-Iris-Dataset-with-scikit-learn/).

Note: This assumes you have basic knowledge of python data science basics. If you don't, or encounter something you're not familiar with, don't worry! You can get a crash course in my guide, [Cleaning MLB Statcast Data using pandas DataFrames and seaborn Visualization](https://chrisman1015.github.io/python/statcast/Cleaning-MLB-Statcast-Data-using-pandas-Dataframes-and-seaborn-Visualization/).

***
t-SNE (t-Distributed Stochastic Neighbor Embedding) is an algorithm for reducing the dimensionality of data primarily for visualization. Essentially, it allows one to see how high-dimensional data is grouped. To see some great examples, check out Laurens van der Maaten's [page](https://lvdmaaten.github.io/tsne/), which has a ton of great example of t-SNE in action.
<br>

This algorithm is particularly useful for gaining quick insight into high dimensional data. t-SNE will transfrom any data into two dimensions quickly and easily. Let's import our basic packages:


```python
# import packages
# datasets has the Iris dataset
from sklearn import datasets

# pandas and numPy for DataFrames and arrays
import pandas as pd
import numpy as np

# pyplot and seaborn for plots
import matplotlib.pyplot as plt
import seaborn as sns

# TSNE Model
from sklearn.manifold import TSNE
```


```python
# load dataset
iris = datasets.load_iris()
```

We'll import the data as usual and create a DataFrame with it:


```python
# load the data and target values
X, y = iris.data, iris.target

# correctly labeled data
iris_labeled = pd.read_csv("labeled_iris.csv", index_col=0)

# create DataFrame with iris data
df = pd.DataFrame(X, columns = iris.feature_names)
df2 = df.copy()
df3 = df.copy()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



We see the iris data has four dimensions: sepal length, sepal width, petal length, and petal width. The t-SNE algorithm will reduce this to two dimensions with no additional information about the data.

Now it's time to intialize and fit the model:


```python
# initialize the model
model = TSNE(learning_rate=100, random_state=2)

# fit the model to the Iris Data
transformed = model.fit_transform(X)
```

The t-SNE algorithm returns a two-dimensional grid of x and y values, each with no scale or dimension:
<br>

To make visualization easier, we'll create a dataframe with these values. We'll then add the target names back in so we can plot it:


```python
df = pd.DataFrame(transformed)
df.columns = ['x', 'y']
df['species'] = iris_labeled['species']
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-16.567120</td>
      <td>-14.909160</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-17.005198</td>
      <td>-17.604240</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-15.914274</td>
      <td>-17.248156</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-16.201323</td>
      <td>-17.750107</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-16.164906</td>
      <td>-14.843614</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



We can use a scatterplot to see what the t-SNE algorithm created:


```python
sns.scatterplot(x='x', y='y', data=df, hue='species',palette="Set1")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18ae5f10ba8>




![png](/assets/images/Unsupervised Learning/t-SNE classification on the Iris Dataset with scikit-learn
/output_11_1.png)


Despite having no information about the target values, the algorithm was able to reduce the dimensionality of the data from 4 to 2, while still clearly differentiating between the species of iris. Remember, it's not the axis of the plot which is important. To illustrate this, we'll run everything again with a different random_state for the model:


```python
# initialize the model
model2 = TSNE(learning_rate=100, random_state=1)
transformed2 = model2.fit_transform(X)

df2 = pd.DataFrame(transformed2)
df2.columns = ['x', 'y']
df2['species'] = iris_labeled['species']

sns.scatterplot(x='x', y='y', data=df2, hue='species',palette="Set1")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18ae5fc0e80>




![png](/assets/images/Unsupervised Learning/t-SNE classification on the Iris Dataset with scikit-learn
/output_13_1.png)

