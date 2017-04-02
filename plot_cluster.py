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
import matplotlib.pyplot as plt

km = kmeans(random_state=5)
gmm = GMM(random_state=5)

indir = './ICA/'

alcohol = pd.read_hdf(indir + 'datasets.hdf','alcohol')
alcoholX = alcohol.drop('Class',1).copy().values
alcoholY = alcohol['Class'].copy().values

iris = pd.read_hdf(indir + 'datasets.hdf', 'iris')
irisX = iris.drop('Class',1).copy().values
irisY = iris['Class'].copy().values

#'Madelon':{'args':{'n_components':30},'cluster':gmm,'name':indir+'madelon2D.csv','dat':madelonX}
clusterplotconfig = {'Alcohol':{'args':{'n_clusters':5},'cluster':km,'name':indir+'alcohol2D.csv','dat':alcoholX},
                    'Iris':{'args':{'n_clusters':3},'cluster':km,'name':indir+'iris2D.csv','dat':irisX},
                     }

for name,d in clusterplotconfig.items():
    pred = d['cluster'].set_params(**d['args']).fit(d['dat']).predict(d['dat'])
    vals = pd.read_csv(d['name'],index_col=0)
    vals['Cluster'] = pred
    vals.plot(kind='scatter',
              x='x',
              y='y',
              c='Cluster',
              title=name + ' k-means with ' + str(d['args']['n_clusters']) + ' clusters',
              cmap='Paired')
    plt.show()