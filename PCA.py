# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:51:37 2017

@author: jtay
"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
import ReliefF as rlf
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM,nn_arch,nn_reg
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

out = './PCA/'
cmap = cm.get_cmap('Spectral') 

np.random.seed(0)
iris = pd.read_hdf('./new_data/datasets.hdf','iris')
irisX = iris.drop('Class', 1).copy().values
irisY = iris['Class'].copy().values

alcohol = pd.read_hdf('./new_data/datasets.hdf', 'alcohol')
alcoholX = alcohol.drop('Class', 1).copy().values
alcoholY = alcohol['Class'].copy().values


alcoholX = StandardScaler().fit_transform(alcoholX)
irisX = StandardScaler().fit_transform(irisX)

clusters =  [2,5,10,15,20,25,30,35,40]
dims = [2,5,10,15,20,25,30]
'''
#raise
#%% data for 1

pca = PCA(random_state=5)
pca.fit(alcoholX)
tmp = pd.Series(data = pca.explained_variance_,index = range(1,33))
tmp.to_csv(out+'alcohol scree.csv')


pca = PCA(random_state=5)
pca.fit(irisX)
tmp = pd.Series(data = pca.explained_variance_,index = range(1,5))
tmp.to_csv(out+'iris scree.csv')


#%% Data for 2

grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='logistic', solver='lbfgs', max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(alcoholX, alcoholY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Alcohol dim red.csv')

dims = [1,2,3]
grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu', solver='lbfgs', max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(irisX, irisY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'iris dim red.csv')

'''
#%% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 30
pca = PCA(n_components=dim,random_state=10)

alcoholX2 = pca.fit_transform(alcoholX)
alcohol2 = pd.DataFrame(np.hstack((alcoholX2,np.atleast_2d(alcoholY).T)))
cols = list(range(alcohol2.shape[1]))
cols[-1] = 'Class'
alcohol2.columns = cols
alcohol2.to_hdf(out+'datasets.hdf','alcohol',complib='blosc',complevel=9)

dim = 3
pca = PCA(n_components=dim,random_state=10)
irisX2 = pca.fit_transform(irisX)
iris2 = pd.DataFrame(np.hstack((irisX2,np.atleast_2d(irisY).T)))
cols = list(range(iris2.shape[1]))
cols[-1] = 'Class'
iris2.columns = cols
iris2.to_hdf(out+'datasets.hdf','iris',complib='blosc',complevel=9)
