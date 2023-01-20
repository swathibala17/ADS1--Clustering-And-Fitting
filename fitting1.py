#!/usr/bin/env python
# coding: utf-8
# ### Fit the Italy data
# 
# The usual imports
# In[26]:
import pandas as pd
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
# import err_ranges as err
# Define the exponential function and the logistics functions for
# fitting.
# In[27]:
#Fittitng
def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    
    f = scale * np.exp(growth * (t-1950)) 
    
    return f
        
def logistics(t, scale, growth, t0):
    """ Computes logistics function with scale, growth raat
    and time of the turning point as free parameters
    """
    
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    
    return f

def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """
    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper        
# In[28]:
# read file with Italy data into dataframe          
df_agriland = pd.read_csv("Agricultural Land.csv")
# have a look
print(df_agriland)
# First fit attempt of the exponential function with defaul initial parameters
# In[29]:
# fit exponential growth
popt, covar = opt.curve_fit(exp_growth, df_agriland["Years"], 
                            df_agriland["Italy"])
# Calculate and plot the result
# In[30]:
print("Fit parameter", popt)
# use *popt to pass on the fit parameters
df_agriland["pop_exp"] = exp_growth(df_agriland["Years"], *popt)
plt.figure()
plt.plot(df_agriland["Years"], df_agriland["Italy"], label="data")
plt.plot(df_agriland["Years"], df_agriland["pop_exp"], label="fit")
plt.legend()
plt.title("First fit attempt")
plt.xlabel("year")
plt.ylabel("Italy")
plt.show()
print()
# Finding a start approximation
# In[31]:
# find a feasible start value the pedestrian way
# the scale factor is way too small. The exponential factor too large. 
# Try scaling with the 1950 Italy and a smaller exponential factor
# decrease or increase exponential factor until rough agreement is reached
# growth of 0.02 gives a reasonable start value
popt = [4e8, 0.01]
df_agriland["pop_exp"] = exp_growth(df_agriland["Years"], *popt)
plt.figure()
plt.plot(df_agriland["Years"], df_agriland["Italy"], label="data")
plt.plot(df_agriland["Years"], df_agriland["pop_exp"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("Italy")
plt.title("Improved start value")
plt.show()
# In[32]:
# fit exponential growth
popt, covar = opt.curve_fit(exp_growth, df_agriland["Years"], 
                            df_agriland["Italy"], p0=[4e8, 0.02])
# much better
print("Fit parameter", popt)
df_agriland["pop_exp"] = exp_growth(df_agriland["Years"], *popt)
plt.figure()
plt.plot(df_agriland["Years"], df_agriland["Italy"], label="data")
plt.plot(df_agriland["Years"], df_agriland["pop_exp"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("Italy")
plt.title("Final fit exponential growth")
plt.show()
print()
# Finding a start approximation for the logistics function
# In[33]:
# estimated turning year: 1990
# Italy in 1990: about 800 million
# kept growth value from before
# increase scale factor and growth rate until rough fit
popt = [8e8, 0.02, 1990]
df_agriland["pop_log"] = logistics(df_agriland["Years"], *popt)
plt.figure()
plt.plot(df_agriland["Years"], df_agriland["Italy"], label="data")
plt.plot(df_agriland["Years"], df_agriland["pop_log"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("Italy")
plt.title("Improved start value")
plt.show()
# In[34]:
popt, covar = opt.curve_fit(logistics, df_agriland["Years"], df_agriland["Italy"], 
                            p0=(2e9, 0.05, 1990.0))
print("Fit parameter", popt)
      
df_agriland["pop_log"] = logistics(df_agriland["Years"], *popt)
plt.figure()
plt.title("logistics function")
plt.plot(df_agriland["Years"], df_agriland["Italy"], label="data")
plt.plot(df_agriland["Years"], df_agriland["pop_log"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("Italy")
plt.show()
# Function from the lecture returning upper and lower limits of the error ranges.
# In[35]:
# extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(covar))
print(sigma)
low, up = err_ranges(df_agriland["Years"], logistics, popt, sigma)
plt.figure()
plt.title("logistics function")
plt.plot(df_agriland["Years"], df_agriland["Italy"], label="data")
plt.plot(df_agriland["Years"], df_agriland["pop_log"], label="fit")
plt.fill_between(df_agriland["Years"], low, up, alpha=0.7)
plt.legend()
plt.xlabel("year")
plt.ylabel("Italy")
plt.show()
# Give ranges
# In[36]:
print("Forcasted Italy")
low, up = err_ranges(2030, logistics, popt, sigma)
print("2030 between ", low, "and", up)
low, up = err_ranges(2040, logistics, popt, sigma)
print("2040 between ", low, "and", up)
low, up = err_ranges(2050, logistics, popt, sigma)
print("2050 between ", low, "and", up)
# Alternative give mean and uncertainty
# In[38]:
print("Forcasted Italy")
low, up = err_ranges(2030, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", mean, "+/-", pm)
low, up = err_ranges(2040, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)
low, up = err_ranges(2050, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2050:", mean, "+/-", pm)
# ## Fit dataset with outliers
# In[39]:
df = pd.read_csv("datapoints.csv")
plt.figure()
plt.scatter(df["x"], df["y"])
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# In[40]:
def poly(x, a, b, c, d):
    """Cubic polynominal for the fitting"""
    
    y = a*x**3 + b*x**2 + c*x + d
    
    return y
param, covar = opt.curve_fit(poly, df["x"], df["y"])
print(param)
# In[41]:
plt.figure()
plt.scatter(df["x"], df["y"])
x = np.linspace(-5.0, 5.0)
plt.plot(x, poly(x, *param), "k")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
print()
# Calculate z-score and remove outliers.
# In[42]:
# produce columns with fit values
df["fit"] = poly(df["x"], *param)
# calculate the z-score
df["diff"] = df["y"] - df["fit"]
sigma = df["diff"].std()
print("Number of points:", len(df["x"]), "std. dev. =", sigma)
# calculate z-score and extract outliers
df["zscore"] = np.abs(df["diff"] / sigma)
df = df[df["zscore"] < 3.0].copy()
# Repeat to check for further outliers
# In[43]:
param, covar = opt.curve_fit(poly, df["x"], df["y"])
# produce columns with fit values
df["fit"] = poly(df["x"], *param)
# calculate the z-score
df["diff"] = df["y"] - df["fit"]
sigma = df["diff"].std()
print("Number of points:", len(df["x"]), "std. dev. =", sigma)
# calculate z-score and extract outliers
df["zscore"] = np.abs(df["diff"] / sigma)
df = df[df["zscore"] < 3.0].copy()
print("Number of points:", len(df["x"]))
# In[44]:
param, covar = opt.curve_fit(poly, df["x"], df["y"])
# produce columns with fit values
df["fit"] = poly(df["x"], *param)
# calculate the z-score
df["diff"] = df["y"] - df["fit"]
sigma = df["diff"].std()
print("Number of points:", len(df["x"]), "std. dev. =", sigma)
# calculate z-score and extract outliers
df["zscore"] = np.abs(df["diff"] / sigma)
df = df[df["zscore"] < 3.0].copy()
print("Number of points:", len(df["x"]))
# In[45]:
plt.figure()
plt.scatter(df["x"], df["y"])
x = np.linspace(-5.0, 5.0)
plt.plot(x, poly(x, *param), "k")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
print()
# In[ ]: