# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:47:56 2017

@author: jtay
"""

import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score as acc
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_selection import mutual_info_classif as MIC
import ReliefF as rlf
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import TransformerMixin,BaseEstimator

nn_arch= [(50,50),(50,),(25,),(25,25),(100,25,100)]
nn_reg = [10**-x for x in range(1,2)]

def cluster_acc(Y,clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
#    assert max(pred) == max(Y)
#    assert min(pred) == min(Y)    
    return acc(Y,pred)


class myGMM(GMM):
    def transform(self,X):
        return self.predict_proba(X)
        
        
def pairwiseDistCorr(X1,X2):
    assert X1.shape[0] == X2.shape[0]
    
    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(),d2.ravel())[0,1]

    
def aveMI(X,Y):    
    MI = MIC(X,Y) 
    return np.nanmean(MI)
    
class reliefSKL(BaseEstimator,TransformerMixin):
    def __init__(self,n_components=10, n_neighbors=100 ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.rlf = rlf.ReliefF(n_features_to_keep=n_components, n_neighbors=n_neighbors)        
    def fit(self,X,y):
        self.rlf.fit(X,y)
        self.feature_scores = self.rlf.feature_scores
        return self
    def fit_transform(self,X,y):        
        return self.rlf.fit_transform(X,y)
    def transform(self,X,y=None):        
        return self.rlf.transform(X)
  
        
# http://datascience.stackexchange.com/questions/6683/feature-selection-using-feature-importances-in-random-forests-with-scikit-learn        
   
class ImportanceSelect(BaseEstimator, TransformerMixin):
    def __init__(self, model, n=1):
         self.model = model
         self.n = n
    def fit(self, *args, **kwargs):
         self.model.fit(*args, **kwargs)
         return self
    def transform(self, X):
         return X[:,self.model.feature_importances_.argsort()[::-1][:self.n]]