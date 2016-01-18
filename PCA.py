#https://www.clear.rice.edu/comp130/12spring/pca/pca_docs.shtml
#http://blog.nextgenetics.net/?e=42


import numpy as np
import matplotlib as plt
from matplotlib.mlab import PCA
import os
import pylab
import sklearn
import gdal
from gdalconst import *
gdal.UseExceptions()

d = '/home/emily/Desktop/MODIS/WF/Drought'
os.chdir(d)

EVIfull = np.load('/media/data/NDVI/MOD13Q1_EVI3D.npy') #still has .min as -.299
EVI = EVIfull[0:230,:,:]
#nanvalue = EVI.min()

#rows as locations and columns as time
EVI2D = np.transpose(EVI.reshape((EVI.shape[0], EVI.shape[1]*EVI.shape[2])))
#EVI2D[EVI2D == nanvalue] = np.nan

#sl mask
SL = gdal.Open('/home/emily/Desktop/MODIS/WF/NDVI_DC/SL.tif', GA_ReadOnly).ReadAsArray() 
SL253 = np.repeat(SL.reshape((1,1927,1082)), EVI.shape[0], axis=0)
SL2D = np.transpose(SL253.reshape((EVI.shape[0], EVI.shape[1]*EVI.shape[2])))

#mask EVI
#EVI2D[SL2D == 0] = np.nan

#pca
myPCA = PCA(EVI2D)

#score matrix (?)
pcDataPoint = myPCA.project(EVI2D) 
PC1 = pcDataPoint[:,1].reshape(EVI.shape[1], EVI.shape[2]) #origin as mean at center of this cluster

#EOFs


#scree
ve = myPCA.fracs
pylab.plot(ve[1:20])
pylab.xlim([0,20]) 

#average for each year
mu = myPCA.mu

#center, scales original data to the center of the cluster and scaled as std dev along new axis
center = myPCA.center #looks identical to the original data

#weights
Wt = myPCA.Wt
#rows are the PC axes

#my data in terms of the principal component axes
Y = myPCA.Y
#PC0 is ocean/land

year = np.arange(0,253,23)
plt.xticks(year)
plt.grid(True)
plt.figure()
plt.plot

PCN = 2
cmap = plt.cm.RdYlGn
cmap.set_bad('w')
plt.figure()
plt.imshow(Y[:,PCN].reshape((1927,1082)), cmap = cmap)
plt.colorbar()


for b in range(0, 15):
	fig = plt.figure(figsize=(7,7))
	data = Y[:,b]
	bound = max(abs(data.min()), data.max())
	plt.imshow(Y[:,b].reshape((1927,1082)), cmap = cmap, vmin = -bound, vmax = bound)
	file_id = str(b).zfill(3) #001
	plt.title('PC' + str(b + 1))
	plt.colorbar()
	plt.savefig(str(d) + '/PCA/' + str(b+1) + '.jpg')            #change directory
	plt.close(fig)


#reshape test
x = np.arange(1,21,1).reshape((5,2,2))
x2D = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
x2DT = x2D.T



#http://glowingpython.blogspot.com/2011/07/principal-component-analysis-with-numpy.html
#scatterplots
plt.plot(Y[:,1], Y[:,2], 'ob')  #is y what i should be plotting, some standardized form
plt.plot(Y[:,4], Y[:,5], 'ob', alpha = 0.03)
axis('equal')

plt.plot(pcDataPoint[:,4], pcDataPoint[:,5], 'ob', alpha = 0.03)


#density scatter
from scipy.stats import gaussian_kde
x = Y[:,3]
y  = Y[:,4]
yc = np.vstack(x,y)
z = gaussian_kde(yc)(yc)
idx = z.argsort()
x,y,z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots()
ax.scatter(x,y,c=z, s=50, edgecolor = '')
plt.show()


	
