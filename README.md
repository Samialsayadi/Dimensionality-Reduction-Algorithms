
# Dimensionality-Reduction-Algorithms
Dimensionality reduction, or dimension reduction, is the transformation of data from a high-dimensional space into a low-dimensional space so that the low-dimensional representation retains some meaningful properties of the original data, ideally close to its intrinsic dimension. Working in high-dimensional spaces can be undesirable for many reasons; raw data are often sparse as a consequence of the curse of dimensionality, and analyzing the data is usually computationally intractable (hard to control or deal with). Dimensionality reduction is common in fields that deal with large numbers of observations and/or large numbers of variables, such as NLP: signal processing, Text classification, speech recognition, neuroinformatics, and bioinformatics.
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
# 3. Principal Component Analysis
# 4. Non-negative Matrix Factorization (NMF)
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







