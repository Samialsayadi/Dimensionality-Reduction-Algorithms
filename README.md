
# Dimensionality-Reduction-Algorithms
Dimensionality reduction, or dimension reduction, is the transformation of data from a high-dimensional space into a low-dimensional space so that the low-dimensional representation retains some meaningful properties of the original data, ideally close to its intrinsic dimension. Working in high-dimensional spaces can be undesirable for many reasons; raw data are often sparse as a consequence of the curse of dimensionality, and analyzing the data is usually computationally intractable (hard to control or deal with). Dimensionality reduction is common in fields that deal with large numbers of observations and/or large numbers of variables, such as NLP: signal processing, Text classification, speech recognition, neuroinformatics, and bioinformatics.

```python
rom sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print("tf-idf with", str(np.array(X_train).shape[1]), "features")
    return (X_train, X_test)
    
Corpus = pd.read_csv(r"aji-Arabic_corpus.csv")
X_train, X_test, y_train, y_test = model_selection.train_test_split(Corpus['text'],Corpus['targe'],test_size=0.2)
X_train, X_test=TFIDF(X_train,X_test)

```

# 1. Random Projection:
random feature is a dimensionality reduction technique mostly used for very large volume dataset or very high dimensional feature space.
```python
from sklearn import random_projection

RandomProjection = random_projection.GaussianRandomProjection(n_components=2000)
X_train_new = RandomProjection.fit_transform(X_train)
X_test_new = RandomProjection.transform(X_test)

print("train with old features: ",np.array(X_train).shape)
print("train with new features:" ,np.array(X_train_new).shape)

print("test with old features: ",np.array(X_test).shape)
print("test with new features:" ,np.array(X_test_new).shape)
```
```
train with old features:  (1200, 45987)
train with new features: (1200, 2000)
test with old features:  (300, 45987)
test with new features: (300, 2000)
```
# 2. Linear Discriminant Analysis (LDA)
Linear Discriminant Analysis (LDA) is another commonly used technique for data classification and dimensionality reduction. LDA is particularly helpful where the within-class frequencies are unequal and their performances have been evaluated on randomly generated test data. Class-dependent and class-independent transformation are two approaches in LDA where the ratio of between-class-variance to within-class-variance and the ratio of the overall-variance to within-class-variance are used respectively.
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA = LinearDiscriminantAnalysis()
X_train_new = LDA.fit(X_train,y_train)
X_train_new =  LDA.transform(X_train)
X_test_new = LDA.transform(X_test)

print("train with old features: ",np.array(X_train).shape)
print("train with new features:" ,np.array(X_train_new).shape)

print("test with old features: ",np.array(X_test).shape)
print("test with new features:" ,np.array(X_test_new).shape)
```
# 3. Principal Component Analysis

Principle component analysis~(PCA) is the most popular technique in multivariate analysis and dimensionality reduction. PCA is a method to identify a subspace in which the data approximately lies. This means finding new variables that are uncorrelated and maximizing the variance to preserve as much variability as possible.
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=1200)
X_train_new = pca.fit_transform(X_train)
X_test_new = pca.transform(X_test)

print("train with old features: ",np.array(X_train).shape)
print("train with new features:" ,np.array(X_train_new).shape)

print("test with old features: ",np.array(X_test).shape)
print("test with new features:" ,np.array(X_test_new).shape)
```
# 4. Non-negative Matrix Factorization (NMF)
Non-negative matrix factorization (NMF or NNMF), also non-negative matrix approximation is a group of algorithms in multivariate analysis and linear algebra where a matrix V is factorized into (usually) two matrices W and H, with the property that all three matrices have no negative elements. This non-negativity makes the resulting matrices easier to inspect. Also, in applications such as processing of audio spectrograms or muscular activity, non-negativity is inherent to the data being considered. Since the problem is not exactly solvable in general, it is commonly approximated numerically.
```python
from sklearn.decomposition import NMF
NMF_ = NMF(n_components=2000)
X_train_new = NMF_.fit(X_train)
X_train_new =  NMF_.transform(X_train)
X_test_new = NMF_.transform(X_test)

print("train with old features: ",np.array(X_train).shape)
print("train with new features:" ,np.array(X_train_new).shape)

print("test with old features: ",np.array(X_test).shape)
print("test with new features:" ,np.array(X_test_new))
```
# 5. Singular Value Decomposition
In linear algebra, the singular value decomposition (SVD) is a factorization of a real or complex matrix. It generalizes the eigendecomposition of a square normal matrix with an orthonormal eigenbasis to any 
m
×
n
m\times n matrix. It is related to the polar decomposition.
```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=1200)
X_train_new = svd.fit(X_train)
X_train_new =  svd.transform(X_train)
X_test_new = svd.transform(X_test)
print("train with old features: ",np.array(X_train).shape)
print("train with new features:" ,np.array(X_train_new).shape)

print("test with old features: ",np.array(X_test).shape)
print("test with new features:" ,np.array(X_test_new))
```
# SVD in NLP (Worked Example) 

```python
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import math
from numpy import linalg as LA

```
#Document-Term-Matrix

 
Dm∗n=USVT
 
where:

U is the doc. concept similarity matrix
V Term concept similarity matrix
S Diagonal element

```python
Doc = np.array([[1, 1,1,0,0], [2, 2,2,0,0],[1, 1,1,0,0],[0, 0,0,2,2],[0, 0,0,3,3],[0, 0,0,1,1]])
Doc
```

# compute matrices
So to find the eigenvalues of the above entity we compute matrices Doc*Doc.T Doc* Doc.^T and Doc.^T* Doc. As previously stated , the eigenvectors of Doc* Doc.^T make up the columns of U so we can do the following analysis to find U.
```python
Doc_U=Doc.dot(Doc.T)
Doc_U
```
 
```python
Doc_v=Doc.T.dot(Doc)
Doc_v
```

# Step 1 find S Diagonal element
```python
results_S_v = la.eig(Doc_v)
results_S_v[0] 
S=np.zeros((6,6), float)
digsig=np.sqrt(results_S_v[0])
indices_diagonal = np.diag_indices(5)

S[indices_diagonal] = digsig
S

```

# Step2: find U the doc. concept similarity matrix


```python
w, U = LA.eig(Doc_U)

print(U)

```
# Step3: find V Term concept similarity matrix

```python
w, VT = LA.eig(Doc_v)

print(VT)
```

# Apply SVD Algorithm

```python
U, s, VT = svd(Doc)
print(U)
print(s)
print(VT)
```
# Dependencies

* [NLTK:](https://anaconda.org/anaconda/nltk) `conda install nltk` 
* [Pandas:](https://pypi.org/project/pandas/) `pip install pandas `.
* [Numpy:](https://anaconda.org/anaconda/numpy) `conda install -c anaconda numpy `.
* [Scikit-cmeans:](https://pypi.org/project/scikit-cmeans) `pip install scikit-cmeans`.






