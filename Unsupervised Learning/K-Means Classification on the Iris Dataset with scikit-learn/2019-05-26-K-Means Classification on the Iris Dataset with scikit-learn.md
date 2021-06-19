---
layout: single
classes: wide
title: "K-Means Classification on the Iris Dataset with scikit-learn"
header:
  teaser: /assets/images/5x3/iris1.png
  overlay_color: "#5e616c"
  overlay_image: /assets/images/1920x1080/iris1.png
  overlay_filter: 0.6
  actions:
  - label: "<i class='fas fa-arrow-alt-circle-right'></i> View in Jupyter"
    url: "https://nbviewer.jupyter.org/github/chrisman1015/Unsupervised-Learning/blob/master/K-Means%20Classification%20on%20the%20Iris%20Dataset%20with%20scikit-learn/K-Means%20Classification%20on%20the%20Iris%20Dataset%20with%20scikit-learn.ipynb"
excerpt: >
  By: Christopher Hauman<br />
categories:
  - Python
  - Unsupervised Machine Learning
tags:
  - Python
  - Machine Learning
  - Unsupervised
  - K-means
  - Classification
---

This will quickly run through using scikit-learn to perform k-means classification on the [Iris](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) dataset. This is a sequel to my guide on [KNN Classification on the Iris Dataset with scikit-learn](https://chrisman1015.github.io//python/supervised%20machine%20learning/KNN-Classification-on-the-Iris-Dataset-with-scikit-learn/). We're also going to build on our previous knowledge by using preprocessing and pipelines to improve our model.

Note: This assumes you have basic knowledge of python data science basics. If you don't, or encounter something you're not familiar with, don't worry! You can get a crash course in my guide, [Cleaning MLB Statcast Data using pandas DataFrames and seaborn Visualization](https://chrisman1015.github.io/python/statcast/Cleaning-MLB-Statcast-Data-using-pandas-Dataframes-and-seaborn-Visualization/).

First, we import some basic packages:


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
```

Import the Iris dataset


```python
# load dataset
iris = datasets.load_iris()
```

Now we'll assign the data and target values into X and y. We'll also create two dataframes with the data lacking the target values. Unlike supervised learning, we won't fit the model with the target values, we'll only use them to evaluate the model afterwards (we'll also cover how to evaluate the model when you don't have any target values).


```python
# load the data and target values
X = iris.data

# create DataFrame with iris data
df = pd.DataFrame(X, columns = iris.feature_names)
df2 = df.copy()
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



Now we import the KMeans package from sklearn and use fit_predict to classify the model. You can also perform the fit and predict steps separately as well. We'll set n_clusters to 3, because we know there's three target species. A bit later we'll cover how to choose a value for n_clusters when you don't know what it should be (which is normal for unsupervised learning).


```python
# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3, random_state=2)

# Fit model to points
y_kmeans = model.fit_predict(X)
print(y_kmeans)
```

    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 2 2 2 1 2 2 2 2
     2 2 1 1 2 2 2 2 1 2 1 2 1 2 2 1 1 2 2 2 2 2 1 2 2 2 2 1 2 2 2 1 2 2 2 1 2
     2 1]
    

We see the KNN classified data into three groups. We can use the **cluster_centers** attribute of the model to find where the clusters are centered. We'll separate them into x and y values for petal and sepal length and width so we can plot them.


```python
centroids = model.cluster_centers_

sep_centroids_x = centroids[:,0]
sep_centroids_y = centroids[:,1]
pet_centroids_x = centroids[:,2]
pet_centroids_y = centroids[:,3]
```

We'll loop through the classified values and add the corresponding name to the DataFrame:


```python
# for loop to name species based on target variable (0='setosa', 1='versicolor', 2='virginica')
names = []
for i in y_kmeans:
    if i == 0:
        names.append("setosa")
    elif i == 1:
        names.append("versicolor")
    else:
        names.append("virginica")
        
# add names to df_labeled2
df["species"] = names
```

Let's import a correctly labeled Iris dataset here so we can also plot that for comparison:


```python
# correctly labeled data
iris_labeled = pd.read_csv("labeled_iris.csv", index_col=0)
iris_labeled.head()
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
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10, 10))

plt.subplot(2,2,1)    
sns.scatterplot(x='sepal length (cm)', y="sepal width (cm)", hue='species', data=df, alpha=0.7)
plt.title("K-means Sepal Length vs Width")
sns.scatterplot(x=sep_centroids_x, y=sep_centroids_y, marker='D', s=50, color='black')

plt.subplot(2,2,2)    
sns.scatterplot(x='sepal length (cm)', y="sepal width (cm)", hue='species', data=iris_labeled)
plt.title("Correctly Labeled Sepal Length vs Width")


plt.subplot(2,2,3) 
sns.scatterplot(x='petal length (cm)', y="petal width (cm)", hue='species', data=df, alpha=0.7)
plt.title("K-means Petal Length vs Width")
sns.scatterplot(x=pet_centroids_x, y=pet_centroids_y, marker='D', s=50, color='black')

plt.subplot(2,2,4) 
sns.scatterplot(x='petal length (cm)', y="petal width (cm)", hue='species', data=iris_labeled)
plt.title("Correctly Labeled Petal Length vs Width")



plt.show()
```


![png](/assets/images/Unsupervised Learning/K-Means Classification on the Iris Dataset with scikit-learn/
output_15_0.png)


We see the algorithm had a much easier time correctly labeling the data using the petals. This is understandable, because the sepals had much more overlap between species. Let's check the accuracy score to see how accurate the model was. We can pull a trick from our supervised learning guide and compute the accuracy score of the model:


```python
# for computing the accuracy of the model.
from sklearn import metrics

print(metrics.accuracy_score(df["species"], iris_labeled["species"]))
```

    0.8933333333333333
    

We see the model correctly classified about 98% of the data.

But how do we evaluate the model when we don't have the correct labels? A simple method is to use the pandas crosstab function to compare the clustered values to the correct labels in a table:


```python
ct = pd.crosstab(y_kmeans, iris_labeled['species'])
print(ct)
```

    species  setosa  versicolor  virginica
    row_0                                 
    0            50           0          0
    1             0          48         14
    2             0           2         36
    

We see the model correctly labeled setosas every time. This makes sense, as the plots show the setosa iris values were grouped far away from the other species for both petal and sepals. Unsuprisingly, the model had much more trouble with versicolor and virginica.

To get a quantitative result, we can use the **inertia** of the model to evaluate it. This measures how tightly grouped the clusters are. The tighter the clusters, the more likely a model is accurate. The K-means algorithm automatically uses inertia to fit the model, and has the inertia value available with the attribute **inertia_**.


```python
print(model.inertia_)
```

    78.85144142614601
    

We see the current model has an inertia of 78.85. You can read more about K-means inertia [here](https://scikit-learn.org/stable/modules/clustering.html).

But how do we choose the number of clusters for the k-means algorithm to fit? In this case we knew how many species of Iris there should be, but we often will not. In this case, we can use an 'elbow plot' of the inertias of the models for different k_values.


```python
ks = range(1, 10)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit_predict(X)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
```


![png](/assets/images/Unsupervised Learning/K-Means Classification on the Iris Dataset with scikit-learn/
output_23_0.png)


A general strategy is to use the number of clusters which corresponds to an 'elbow' in the plot, or where the slope stops decreasing quickly. In this case, we see the slope levels off sharply at 3 clusters, which is the value we chose. The reason for this approach is that the elbow corresponds to the best value to balance the risk of overfitting and underfitting. Note that choosing two clusters also would've yielded a decent inertia, as we discussed how similar the virginica and cersicolor data can be. If we only cared about inertia, we could chose a k value equal to the size of the dataset, where each point becomes a cluster. The inertia would become 0, but we'd have no useful information:


```python
# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model_bad = KMeans(n_clusters=len(X)-1)

# Fit model to points
y_kmeans = model_bad.fit_predict(X)
print(model_bad.inertia_)
```

    0.0
    

The point of clustering is to detect patterns, and we lose that by overfitting the model. This was simply to illustrate the usefulness of the elbow plot. We'll usually perform the elbow plot much earlier in the process, before we commit to a k value.

The last thing worth touching on is using a tranformation to increase the efficacy of the model. Many machine learning models require some form of preprocessing to work correctly, and it's often extremely helpful when dealing with data with different variances or scales. You can read much more about preprocessing [here](https://scikit-learn.org/stable/modules/preprocessing.html).

In this case, we'll perfom a normalization transformation on the data before clustering it with k-means. Normalization will transform each sample (row) in the data so they're on the same scale. This will ensure that each sample has an equal say in the process of classifying the data. This is not to be confused with [StandardScaler](https://stackoverflow.com/questions/39120942/difference-between-standardscaler-and-normalizer-in-sklearn-preprocessing), which is also quite useful. We'll start by importing the Normalizer and sklearn's pipeline functionality:


```python
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
```

To perform two steps, we initalize each of the algorithms and then feed them into a pipeline. The pipeline can then act as a model on the data which will perform each algorithm in the pipeline sequentially:


```python
# initialize the algorithms
normalizer = Normalizer()
kmeans = KMeans(n_clusters=3, random_state=10)

# create the pipeline and fit the data
pipeline = make_pipeline(normalizer, kmeans)
normalized = pipeline.fit_predict(X)

# display the classified values
normalized
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])



Let's compare the crosstab tables of the normalized and non-normalized classified data:


```python
ct2 = pd.crosstab(normalized, iris_labeled['species'])
print("The Crosstab for the Normalized K-Means:")
print(ct2)
print("\n\nThe Crosstab for the non-normalized K-Means:")
print(ct)
```

    The Crosstab for the Normalized K-Means:
    species  setosa  versicolor  virginica
    row_0                                 
    0            50           0          0
    1             0          45          0
    2             0           5         50
    
    
    The Crosstab for the non-normalized K-Means:
    species  setosa  versicolor  virginica
    row_0                                 
    0            50           0          0
    1             0          48         14
    2             0           2         36
    

We see a significant improvement in the normalized data classification for the versicolor and virginica data. The normalized model correctly identified all but five data points, compared to 16 for the non-normalized model.

We'll perform the loop again to add the names to the second copy of df we made so we can get ready to do our last set of plots:


```python
# for loop to name species based on target variable (0='setosa', 1='versicolor', 2='virginica')
names = []
for i in normalized:
    if i == 0:
        names.append("setosa")
    elif i == 1:
        names.append("versicolor")
    else:
        names.append("virginica")
        
# add names to df2
df2["species"] = names

#h get centroids
centroids2 = pipeline.steps[1][1].cluster_centers_
sep_centroids_x2 = centroids[:,0]
sep_centroids_y2 = centroids[:,1]
pet_centroids_x2 = centroids[:,2]
pet_centroids_y2 = centroids[:,3]
```

Note that we had to access the specific step of the pipeline which contained the K-means algorithm in order to access the attribute for cluster centers. You can read more about that in the [sklearn documentation](https://scikit-learn.org/dev/modules/compose.html) under "Accessing Steps."

Before the plots, let's print out the intertias and accuracy scores for the two models to make some final quantitative comparisons.


```python
old_acc = metrics.accuracy_score(df["species"], iris_labeled["species"])
new_acc = metrics.accuracy_score(df2["species"], iris_labeled["species"])

percent_change_intertia = ( (pipeline.steps[1][1].inertia_) - (model.inertia_) ) / (model.inertia_) * 100
percent_change_accuracy = ( (new_acc) - (old_acc) ) / (old_acc) * 100

print("The inertia for the normalized model is: " + str(pipeline.steps[1][1].inertia_) +
      "\nThe inertia for the non-normalized model is: " + str(model.inertia_) + 
      "\nPercent change in inertia: " + str(percent_change_intertia))

print("\n\nThe accuracy score for the normalized model is: " + str(new_acc) +
      "\nThe accuracy score for the non-normalized model is: " + str(old_acc) + 
      "\nPercent change in accuracy score: " + str(percent_change_accuracy))
```

    The inertia for the normalized model is: 0.32268174048328563
    The inertia for the non-normalized model is: 28.27375008667915
    Percent change in inertia: -98.85872323447
    
    
    The accuracy score for the normalized model is: 0.9666666666666667
    The accuracy score for the non-normalized model is: 0.8933333333333333
    Percent change in accuracy score: 8.2089552238806


That's a huge difference! By simply normalizing the observations in our data, k-means was able to model the data with over 8% higher accuracy with almost half the intertia. This is only two percent less accurate than our supervised k-nn model which used gridseach to optimize the hyperparameters! It's amazing how accurate these models can be, even when it has no target values to learn from! Remember the key to this result was our preprocessing. You will often need to try different algorithms, hyperparameters, and preprocessing methods to optimize your model.


Finally, We'll plot the normalized classified data alongside the non-normalized data and the correct data:


```python
plt.figure(figsize=(15, 10))

plt.subplot(2,3,1)    
sns.scatterplot(x='sepal length (cm)', y="sepal width (cm)", hue='species', data=df2, alpha=0.7)
plt.title("Normalized K-means Sepal Length vs Width")
sns.scatterplot(x=sep_centroids_x2, y=sep_centroids_y2, marker='D', s=50, color='black')

plt.subplot(2,3,2)    
sns.scatterplot(x='sepal length (cm)', y="sepal width (cm)", hue='species', data=df, alpha=0.7)
plt.title("non-Normalized K-means Sepal Length vs Width")
sns.scatterplot(x=sep_centroids_x, y=sep_centroids_y, marker='D', s=50, color='black')

plt.subplot(2,3,3)    
sns.scatterplot(x='sepal length (cm)', y="sepal width (cm)", hue='species', data=iris_labeled)
plt.title("Correctly Labeled Sepal Length vs Width")


plt.subplot(2,3,4) 
sns.scatterplot(x='petal length (cm)', y="petal width (cm)", hue='species', data=df2, alpha=0.7)
plt.title("Normalized K-means Petal Length vs Width")
sns.scatterplot(x=pet_centroids_x2, y=pet_centroids_y2, marker='D', s=50, color='black')

plt.subplot(2,3,5) 
sns.scatterplot(x='petal length (cm)', y="petal width (cm)", hue='species', data=df, alpha=0.7)
plt.title("non-Normalized K-means Petal Length vs Width")
sns.scatterplot(x=pet_centroids_x, y=pet_centroids_y, marker='D', s=50, color='black')

plt.subplot(2,3,6) 
sns.scatterplot(x='petal length (cm)', y="petal width (cm)", hue='species', data=iris_labeled)
plt.title("Correctly Labeled Petal Length vs Width")

plt.show()
```


![png](/assets/images/Unsupervised Learning/K-Means Classification on the Iris Dataset with scikit-learn/
output_37_0.png)


It's amazing how well the normalized model was able to classify the data despite the overlap in the sepal data for virginicas and versicolors.

That does it! This guide was inspired by (and some bit of code were adapted from) Datacamp's course on [Unsupervised Learning in Python](https://www.datacamp.com/courses/unsupervised-learning-in-python). It's a great resource for learning all about data science and machine learning!
