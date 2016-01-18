# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:19:54 2015

@author: emily
"""

import numpy as np
import numpy.ma as ma
import pylab as plt
import gdal
from gdalconst import *
gdal.UseExceptions()
import os
from scipy import interpolate
import sg
from sg import *

d = '/home/emily/Desktop/MODIS/WF/Drought'
os.chdir(d)
data = '/media/data/NDVI/columns/MOD13Q1_EVI.npy'
shapemask = '/home/emily/Desktop/MODIS/WF/NDVI_DC_2_18/PADDY.tif'
band = 253 #253
threshold = 151 #60 percent
nanvalue = -9999.0
row = 1927
col = 1082


EVI = np.load(data).reshape((band, row, col))
PAD = gdal.Open(shapemask, GA_ReadOnly).ReadAsArray()
PADr = np.tile(PAD, (band, 1, 1))

EVImask = ma.masked_where(PADr!=1, EVI)
EVIm = ma.masked_where(EVI == nanvalue, EVImask)
del EVImask


#count non-masked elements
count = EVIm.count(axis=0)
#np.histogram(count[count<56], bins=100)

#mask array where the number of non-masked elements is less than the threshold 
pixels = ma.masked_where(count < threshold, count).reshape((1, row, col))
pixelsr = np.tile(pixels, (band, 1, 1))

EVImask = ma.masked_where(pixelsr == True, EVIm)
#pixels are the pixels of interest, EVIm is the masked datacube 

plt.imshow(EVImask[22,:,:])
plt.colorbar()

#code works up to here

#interpolate
EVIint = np.empty((band, row, col))
x = np.arange(253.)

out = np.ma.filled(pixels, nanvalue)

for r in xrange(EVImask.shape[1]):
	for c in xrange(EVImask.shape[2]):
		if out[0,r,c] != nanvalue:
			pixel = EVImask[:,r,c]
			y_ = pixel
			x_ = (np.arange(len(y_)))[~pixel.mask] #index of pixels with good data
			y_ = y_[~pixel.mask] #pulls out good values
			f = interpolate.interp1d(x_, y_, kind = 'linear', bounds_error = False, fill_value = nanvalue)
			ynew = f(x)
			EVIint[:,r,c] = ynew  
		if out[0, r,c] == nanvalue:
			EVIint[:,r,c] = nanvalue
		




#Chen et al.
#test with one pixel

ws1 = 11 #sg1
o1 =  6
ws2 = 11 #sg2 (m = 4, d = 6)
o2 = 6
#np.where(~EVImask.mask)
r = 1785
c = 651
year = np.arange(0,253,23)
original = EVImask[:,r,c]
x = np.arange(len(pixel))

#1 - interpolate
xmask = (np.arange(len(original)))[~original.mask]
ymask = original[~original.mask]
f = interpolate.interp1d(xmask, ymask, kind = 'linear')
interp = f(x)
#EVI int already has all pixels inteprolated

#2. sg1
sg = savitzky_golay(interp, ws1, o1) #find m and d using least-squares fitting method of all combinations

#3. weights
weight = np.zeros_like(sg)
dmax = max(abs(interp - sg))

for i in range(0,len(sg)):
	if interp[i] >= sg[i]:
		weight[i] = 1
	if interp[i] < sg[i]:
		weight[i] = 1 - (abs(interp[i] - sg[i])/dmax)
		
		
#########################################			
#iterations
#########################################

num_it = 100
obs = 253
EVIsg = np.zeros_like(EVI)

#for r in range(0, row):
	#for c in range(0, col):
		#first sg
		#interpolation

fi = np.zeros(num_it)
#sgn = sg.reshape((obs, 1))
sgn = []
sgn.append(sg) #includes first sg for pixel
finalpix = np.zeros((obs,1))

for it in range(0,len(fi)):
	sgn_array = np.transpose(np.asarray(sgn)) #worked
	
	newts = np.zeros(obs)
	for i in range(0,obs):
		if interp[i] >= sgn_array[i,it]:
			newts[i] = interp[i]
		if interp[i] < sgn_array[i,it]:
			newts[i] = sgn_array[i,it]
		
		fipix = (abs(sgn_array[i,it] - interp[i]))*weight[i]
		
	sg = savitzky_golay(newts, ws2, o2)
	sgn.append(sg)
	
	fi[it] += fipix	
	
	stop = find(min(fi))
	EVIsg[:,r,c] = sgn[stop[0]]
	
	#if fi[it] > fi[it-1] < fi[it-2]:
		#evetually, index full array and replace dataset with sgn[:.it-1]
		#finalpix[:,0] = np.asarray(sgn[it-1])
		#np.append(finalpix, sgn[:,it-1])
	
		
#works like a dream except for exit condition, b/c no minima in dataset
#iterations just push upper envelope



#plotting

r = 1282
c = 548

pixel = EVImask[:,r,c]
x = range(0,253)
ynew = 

def applyPlotStyle():
	plt.ylabel('EVI')
	plt.xticks(year)
	plt.grid(True)

plt.subplot(3,1,1)
applyPlotStyle()
plt.plot(x, pixel , 'r*')
plt.plot(x, pixel, 'k--')
plt.title('original data')

plt.subplot(3,1,2)
applyPlotStyle()
plt.plot(xnew, ynew)
plt.title('linear interpolation')

plt.subplot(3,1,3)
applyPlotStyle()
plt.plot(xnew,ynew,'k--',label='interp')
plt.plot(xnew,z, 'm', label = 'sg')
#plt.xlim(x[0], x[-1])
plt.legend(loc = 'best')
plt.title('smoothing filter width %d order %.2f'%(window_size, order))
plt.xlabel('Time (16-day)')





#archive

y = pixel
x_ = (np.arange(len(y)))[~pixel.mask]
y_ = y_[~pixel.mask]
xnew = np.arange(1.,253.)
f = interpolate.interp1d(x_, y_, kind = 'linear')
ynew = f(xnew)
plt.subplot(3,1,2)
applyPlotStyle()
plt.plot(xnew, ynew)
plt.title('linear interpolation')


#Chen et al.
#test with one pixel

ws1 = 11 #sg1
o1 =  6
ws2 = 9 #sg2 (m = 4, d = 6)
o2 = 6
#np.where(~EVImask.mask)
r = 1283
c = 520
year = np.arange(0,253,23)
original = EVImask[:,r,c]
x = np.arange(len(pixel))

#1 - interpolate
xmask = (np.arange(len(original)))[~original.mask]
ymask = original[~original.mask]
f = interpolate.interp1d(xmask, ymask, kind = 'linear')
interp = f(x)

#2. sg1
sg = savitzky_golay(interp, ws1, o1) #find m and d using least-squares fitting method of all combinations

#3. weights
weight = np.zeros_like(sg)
dmax = max(abs(interp - sg))

for i in range(0,len(sg)):
	if interp[i] >= sg[i]:
		weight[i] = 1
	if interp[i] <= sg[i]:
		weight[i] = 1 - (abs(interp[i] - sg[i])/dmax)
			

#4. create new time series
newts = np.zeros_like(sg)

for i in range(0,len(sg)):
	if interp[i] >= sg[i]:
		newts[i] = interp[i]
	if interp[i] <= sg[i]:
		newts[i] = sg[i]

#5. SG on newts
sgn = savitzky_golay(newts, ws2, o2)
		
#6 fitting index
fi = 0
for i in range(0, len(sg1)):
	fipix = (sgn[i] - interp[i])*weight[i]
	fi += fipix





###working iteration except for FI exit


num_it = 100
obs = 253
EVIsg = np.zeros_like(EVI)

#for r in range(0, row):
	#for c in range(0, col):
		#first sg
		#interpolation

fi = np.zeros(num_it)
#sgn = sg.reshape((obs, 1))
sgn = []
sgn.append(sg) #includes first sg for pixel
finalpix = np.zeros((obs,1))

for it in range(0,len(fi)):
	sgn_array = np.transpose(np.asarray(sgn)) #worked
	
	newts = np.zeros(obs)
	for i in range(0,obs):
		if interp[i] >= sgn_array[i,it]:
			newts[i] = interp[i]
		if interp[i] < sgn_array[i,it]:
			newts[i] = sgn_array[i,it]
		
		fipix = (abs(sgn_array[i,it] - interp[i]))*weight[i]
		
	sg = savitzky_golay(newts, ws2, o2)
	sgn.append(sg)
	
	fi[it] += fipix	
	
	
	
	if fi[it] > fi[it-1] < fi[it-2]:
		#evetually, index full array and replace dataset with sgn[:.it-1]
		finalpix[:,0] = np.asarray(sgn[it-1])
		#np.append(finalpix, sgn[:,it-1])
	
		
#works like a dream except for exit condition, b/c no minima in dataset
#iterations just push upper envelope
