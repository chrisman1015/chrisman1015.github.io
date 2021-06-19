---
layout: single
classes: wide
title: "TruncatedSVD Decomposition and K-Means Classification on tf-idf Data with scikit-learn"
header:
  teaser: /assets/images/5x3/wrench.png
  overlay_color: "#5e616c"
  overlay_image: /assets/images/1920x1080/wrench.png
  overlay_filter: 0.6
  actions:
  - label: "<i class='fas fa-arrow-alt-circle-right'></i> View in Jupyter"
    url: "https://nbviewer.jupyter.org/github/chrisman1015/Unsupervised-Learning/blob/master/TruncatedSVD%20Decomposition%20and%20K-Means%20Classification%20on%20tf-idf%20Data%20with%20scikit-learn/K-Means%20Classification%20on%20tf-idf%20Data%20with%20TruncatedSVD%20and%20scikit-learn.ipynb"
excerpt: >
  By: Christopher Hauman<br />
categories:
  - Python
  - Unsupervised Machine Learning
tags:
  - Python
  - Machine Learning
  - Unsupervised
  - PCA
  - TruncatedSVD
  - Text
---
This is a sequel to my guides on [K-Means Classification on the Iris Dataset](https://chrisman1015.github.io//python/unsupervised%20machine%20learning/K-Means-Classification-on-the-Iris-Dataset-with-scikit-learn/) and [Principal Component Analysis on the Iris Dataset](https://chrisman1015.github.io//python/unsupervised%20machine%20learning/Principal-Component-Analysis/).

Note: This assumes you have basic knowledge of python data science basics. If you don't, or encounter something you're not familiar with, don't worry! You can get a crash course in my guide, [Cleaning MLB Statcast Data using pandas DataFrames and seaborn Visualization](https://chrisman1015.github.io/python/statcast/Cleaning-MLB-Statcast-Data-using-pandas-Dataframes-and-seaborn-Visualization/).

***
To begin with, we're going to introduce the concept of tf-idf data. tf-idf stands for 'term frequency-inverse document frequency' and the data consists of weighted frequency values. To illustrate this, we'll explain the example we're working with.
<br>

The tf-idf data we're going to work with is the weighted frequency of words in a set of Wikipedia articles. We have 60 articles, 13,125 words, and the entries of the tf-idf array are the weighted frequency of that word's appearence in the article in comparison to the rest of the articles. For instance, if the word 'the' appears in the article "Internet Explorer" 200 times, it would likely still have a lower value in the tf-idf table than the word 'Thomas' (the original author of IE) because 'the' appears many time in every article, while 'Thomas' may only appear in that single article.
<br>

As you can imagine, tf-idf has a lot of zero entries. If you take a look at the wikipedia-vocabulary text file, you should see why this is the case. 'Thomas' probably only has a nonzero value for the Internet Explorer article. Because of this, we often store tf-idf data in a [CSR matrix](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html). This is a type of sparse matrix (meaning there aren't many nonzero values in the matrix) which saves memory by only storing the value for non-zero entries. This saves a lot of memory, but unfortunately isn't compatible with PCA. Fortunately, there's another option which performs the same operation as PCA on CSR matrices, called TruncatedSVD.

We're also going to crank the modeling up a notch by pairing the dimensionality reduction of TruncatedSVD with K-means classification. We'll see how the model uses the tf-idf data to classify it!


```python
# import libraries
import pandas as pd
from scipy.sparse import csr_matrix

# import data
df = pd.read_csv('wikipedia-vectors.csv', index_col=0)
articles = csr_matrix(df.transpose())
titles = list(df.columns)
```

Similar to in other guides, we'll create a pipeline which applies the TruncatedSVD algorithm to the data followed by K-means classification:


```python
# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=20)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=7)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)
```

Now that we've made the model, let's fit it to the tf-idf data (stored in the articles CSR matrix) and make the classification prediction in one step:


```python
labels = pipeline.fit_predict(articles)
```

We'll store the classified labels and values in a DataFrame and print them to see how the model classified them:


```python
# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# print clusters
for i in range(6):
    print("\n\nCluster " + str(i) + "\n")
    print(df[df['label'] == i])
```

    
    
    Cluster 0
    
        label           article
    10      0    Global warming
    12      0      Nigel Lawson
    13      0  Connie Hedegaard
    14      0    Climate change
    16      0           350.org
    
    
    Cluster 1
    
        label               article
    20      1        Angelina Jolie
    21      1    Michael Fassbender
    22      1     Denzel Washington
    23      1  Catherine Zeta-Jones
    24      1          Jessica Biel
    25      1         Russell Crowe
    26      1            Mila Kunis
    27      1        Dakota Fanning
    28      1         Anne Hathaway
    29      1      Jennifer Aniston
    
    
    Cluster 2
    
        label      article
    40      2  Tonsillitis
    41      2  Hepatitis B
    42      2  Doxycycline
    43      2     Leukemia
    44      2         Gout
    45      2  Hepatitis C
    46      2   Prednisone
    47      2        Fever
    48      2   Gabapentin
    49      2     Lymphoma
    
    
    Cluster 3
    
        label                article
    50      3           Chad Kroeger
    51      3             Nate Ruess
    52      3             The Wanted
    53      3           Stevie Nicks
    54      3         Arctic Monkeys
    55      3          Black Sabbath
    56      3               Skrillex
    57      3  Red Hot Chili Peppers
    58      3                 Sepsis
    59      3            Adam Levine
    
    
    Cluster 4
    
       label                      article
    0      4                     HTTP 404
    1      4               Alexa Internet
    2      4            Internet Explorer
    3      4                  HTTP cookie
    4      4                Google Search
    5      4                       Tumblr
    6      4  Hypertext Transfer Protocol
    7      4                Social search
    8      4                      Firefox
    9      4                     LinkedIn
    
    
    Cluster 5
    
        label                            article
    30      5      France national football team
    31      5                  Cristiano Ronaldo
    32      5                       Arsenal F.C.
    33      5                     Radamel Falcao
    34      5                 Zlatan Ibrahimović
    35      5    Colombia national football team
    36      5  2014 FIFA World Cup qualification
    37      5                           Football
    38      5                             Neymar
    39      5                      Franck Ribéry
    

With no instruction on what categories to make, the model was able to classify actors, musical artists, sports-related articles, and make other groupings easily and accurately.

Though we couldn't use PCA, we were able to perform the same operation on the CSR matrix with truncatedSVD. If you want to take the next step and learn how to build a recommendation engine in python on tf-idf data, check out my guide on [NMF Decomposition and K-Means Clustering on tf-idf Wikipedia Text with scikit-learn](https://nbviewer.jupyter.org/github/chrisman1015/Unsupervised-Learning/blob/master/NMF%20Decomposition%20and%20K-Means%20Clustering%20on%20tf-idf%20Wikipedia%20Text%20with%20scikit-learn/NMF%20Decomposition%20and%20K-Means%20Clustering%20on%20tf-idf%20Wikipedia%20Text%20with%20scikit-learn.ipynb).

The tf-idf data originated [here](https://blog.lateral.io/2015/06/the-unknown-perils-of-mining-wikipedia/) and the code is adapted from Datacamp's course on [Unsupervised Learning](https://www.datacamp.com/courses/unsupervised-learning-in-python).
