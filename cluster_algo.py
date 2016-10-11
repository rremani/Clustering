# clustering based on distance matrix
from __future__ import division
import numpy as np
#import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance
# Taking lat long of delhi for the example
lat = 28.7041
lon = 77.1025

# Radius which is equal to 100 kms since one degree is equal to 111300 m
r = 100000/111300 
np.random.seed(0)
# Generate random 
y0= lat
x0= lon

u = np.random.random_sample(100)
v = np.random.random_sample(100)

w = r * np.sqrt(u)
t = 2 * np.pi * v

x = w * np.cos(t)
y1 = w * np.sin(t)
x1 = x / np.cos(y0)

newY = y0 + y1
newX = x0 + x1
 
coord = np.radians(np.transpose(np.array([newY,newX])))*6371
dist  = distance.cdist(coord,coord,'euclidean')
values = np.random.random_integers(1, 100, 100)
ids = np.arange(100)
capacity = np.mean(values)

for i in range(100):
	k = dist[i,:]
	a = k < 50 
	k = k[a]
	values_new = values[a]
	ids_new = ids[a]
	sum_v = np.sum(values_new)
	sum_v = sum_v - values[i]
	if sum_v < capacity:
		ids_new[:] = ids[i]
		ids[a]=ids_new 
	
	
print values,ids
print len(np.unique(ids))
plt.plot(newY,newX,"ro")
plt.show()