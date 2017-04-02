# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:38:28 2017

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
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM,nn_arch,nn_reg
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys

out = './{}/'.format(sys.argv[1])

np.random.seed(0)
cmap = cm.get_cmap('Spectral') 
alcohol = pd.read_hdf(out+'datasets.hdf','alcohol')
alcoholX = alcohol.drop('Class',1).copy().values
alcoholY = alcohol['Class'].copy().values

iris = pd.read_hdf(out + 'datasets.hdf', 'iris')
irisX = iris.drop('Class',1).copy().values
irisY = iris['Class'].copy().values


alcoholX = StandardScaler().fit_transform(alcoholX)
irixX= StandardScaler().fit_transform(irisX)

clusters =  [2,5,10,15,20,25,30,35,40]

#%% Data for 1-3
SSE = defaultdict(dict)
ll = defaultdict(dict)
acc = defaultdict(lambda: defaultdict(dict))
adjMI = defaultdict(lambda: defaultdict(dict))
km = kmeans(random_state=5)
gmm = GMM(random_state=5)

st = clock()
for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)
    km.fit(irisX)
    gmm.fit(irisX)
    SSE[k]['Iris'] = km.score(irisX)
    ll[k]['Iris'] = gmm.score(irisX)
    acc[k]['Iris']['Kmeans'] = cluster_acc(irisY,km.predict(irisX))
    acc[k]['Iris']['GMM'] = cluster_acc(irisY,gmm.predict(irisX))
    adjMI[k]['Iris']['Kmeans'] = ami(irisY,km.predict(irisX))
    adjMI[k]['Iris']['GMM'] = ami(irisY,gmm.predict(irisX))
    
    km.fit(alcoholX)
    gmm.fit(alcoholX)
    SSE[k]['Alcohol'] = km.score(alcoholX)
    ll[k]['Alcohol'] = gmm.score(alcoholX)
    acc[k]['Alcohol']['Kmeans'] = cluster_acc(alcoholY,km.predict(alcoholX))
    acc[k]['Alcohol']['GMM'] = cluster_acc(alcoholY,gmm.predict(alcoholX))
    adjMI[k]['Alcohol']['Kmeans'] = ami(alcoholY,km.predict(alcoholX))
    adjMI[k]['Alcohol']['GMM'] = ami(alcoholY,gmm.predict(alcoholX))
    print(k, clock()-st)
    
    
SSE = (-pd.DataFrame(SSE)).T
SSE.rename(columns = lambda x: x+' SSE (left)',inplace=True)
ll = pd.DataFrame(ll).T
ll.rename(columns = lambda x: x+' log-likelihood',inplace=True)
acc = pd.Panel(acc)
adjMI = pd.Panel(adjMI)


SSE.to_csv(out+'SSE.csv')
ll.to_csv(out+'logliklihood.csv')
acc.ix[:,:,'Alcohol'].to_csv(out+'Alcohol acc.csv')
acc.ix[:,:,'Iris'].to_csv(out+'Iris acc.csv')
adjMI.ix[:,:,'Alcohol'].to_csv(out+'Alcohol adjMI.csv')
adjMI.ix[:,:,'Iris'].to_csv(out+'Iris adjMI.csv')


#%% NN fit data (2,3)

grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(solver='lbfgs',activation='logistic',max_iter=2000,early_stopping=True,random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km',km),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10)

#gs.fit(madelonX,madelonY)
#tmp = pd.DataFrame(gs.cv_results_)
#tmp.to_csv(out+'Madelon cluster Kmeans.csv')


grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='logistic', solver='lbfgs', max_iter=2000, early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

#gs.fit(madelonX,madelonY)
#tmp = pd.DataFrame(gs.cv_results_)
#tmp.to_csv(out+'Madelon cluster GMM.csv')




grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='logistic', solver='lbfgs', max_iter=2000, early_stopping=True,random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km',km),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(alcoholX,alcoholY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Alcohol cluster Kmeans.csv')


grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='logistic', solver='lbfgs', max_iter=2000, early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(alcoholX,alcoholY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Alcohol cluster GMM.csv')


# %% For chart 4/5
irisX2D = TSNE(verbose=10,random_state=5).fit_transform(irisX)
alcoholX2D = TSNE(verbose=10,random_state=5).fit_transform(alcoholX)

iris2D = pd.DataFrame(np.hstack((irisX2D,np.atleast_2d(irisY).T)),columns=['x','y','target'])
alcohol2D = pd.DataFrame(np.hstack((alcoholX2D,np.atleast_2d(alcoholY).T)),columns=['x','y','target'])

iris2D.to_csv(out+'iris2D.csv')
alcohol2D.to_csv(out+'alcohol2D.csv')


