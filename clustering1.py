# In[1]:
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 09:54:28 2021
Clustering exercise using fish measurements
@author: napi
"""
import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import scipy.optimize as opt
# get_ipython().run_line_magic('matplotlib', 'inline')
# Define functions to normalise one array and iterate over all numerical columns of the dataframe
# In[2]:
#----------------------------------------------
def heat_corr(df, size=10):
    """Function creates heatmap of correlation matrix for each pair of columns 
    in the dataframe.
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()
    
    return
#-------------------------------------------------
    
def norm(array):
    """ Returns array normalised to [0,1]. Array can be a numpy array 
    or a column of a dataframe"""
    min_val = np.min(array) 
    max_val = np.max(array)
    
    scaled = (array-min_val) / (max_val-min_val)  
    
    return scaled
def norm_df(df, first=0, last=None):
    """ 
    Returns all columns of the dataframe normalised to [0,1] with the 
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    First, last: columns from first to last (including) are normalised.
    Defaulted to all. None is the empty entry. The default corresponds 
    """
    
    # iterate over all numerical columns
    for col in df.columns[first:last]:     # excluding the first column
        df[col] = norm(df[col])
        
    return df
# Read in and inspect
# In[3]:
# reading the file and basic statistics
df_agriland = pd.read_csv("Agricultural Land.csv", skiprows=(1,2))
df_agriland=df_agriland.iloc[1:30,1:30]
print(df_agriland.describe())
print(df_agriland)
# Display heatmap and scatter plot.
# 
# `coolwarm` and other colour maps.
# ![Colour maps](div_colormaps.png)
# 
# Combinations of columns with lighjt blue or light red are good.
# In[4]:
# heatmap
heat_corr(df_agriland, 9)
# Total length vs. height splits has a low-ish correlation. Picking that combination. The scatter plot confirms that this is a good choice. 
# In[5]:
pd.plotting.scatter_matrix(df_agriland, figsize=(9.0, 9.0))
plt.tight_layout()    # helps to avoid overlap of labels
plt.show()
# Setting up and executing kmeans clustering. Running a loop iterating the number of clusters and calculate the silouette score.
# In[6]:
# extract columns for fitting
df_fit = df_agriland[["Burundi","Benin","Brazil","Costa Rica","Cyprus","Haiti","Italy","Sri Lanka",	"Maldives",	"Malaysia",	"Nigeria"]].copy()
# normalise dataframe and inspect result
# normalisation is done only on the extract columns. .copy() prevents
# changes in df_fit to affect df_agriland. This make the plots with the 
# original measurements
df_fit = norm_df(df_fit)
print(df_fit.describe())
print()
for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)     
    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_fit, labels))
    
    
# Good results for 4 and 5 clusters. Plot both. Note that plt.scatter() enabler a more elegant method to colour code the symbols than the loop in the lecture.
# In[7]:
# Plot for four clusters
kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(df_fit)     
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
# Individual colours can be assigned to symbols. The label l is used to the select the 
# l-th number from the colour table.
plt.scatter(df_fit["Benin"], df_fit["Nigeria"], c=labels, cmap="Accent")
# colour map Accent selected to increase contrast between colours
# show cluster centres
for ic in range(4):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
    
plt.xlabel("Benin")
plt.ylabel("Nigeria")
plt.title("4 clusters")
plt.show()

#___________________________________________________________________________
 
