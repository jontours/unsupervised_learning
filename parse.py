# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:39:27 2017

@author: jtay
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


d = defaultdict(LabelEncoder)
OUT = './new_data/'

dataframe = pd.read_csv(filepath_or_buffer="./data/student-por.csv", sep=';')
dataframe1 = pd.read_csv(filepath_or_buffer="./data/student-mat.csv", sep=';')
dfs = [dataframe, dataframe1]
df = pd.concat(dfs)
df = df.drop_duplicates()

alcohol = df.apply(lambda x: d[x.name].fit_transform(x))
alcohol.rename(columns = {'Walc':'Class'}, inplace = True)
#cols = list(range(alcohol.shape[1]))
#cols['Walc'] = 'Class'
#alcohol.columns = cols
alcohol.to_hdf(OUT+'datasets.hdf','alcohol',complib='blosc',complevel=9)

iris = load_iris(return_X_y=True)
irisX,irisY = iris

iris = np.hstack((irisX, np.atleast_2d(irisY).T))
iris = pd.DataFrame(iris)
cols = list(range(iris.shape[1]))
cols[-1] = 'Class'
iris.columns = cols
iris.to_hdf(OUT+'datasets.hdf','iris',complib='blosc',complevel=9)