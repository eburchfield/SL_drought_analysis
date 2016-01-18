sav
import numpy as np
import matplotlib as plt
from matplotlib.mlab import PCA
import scipy
from scipy import integrate

Y = np.load('/data/emily/WF/Drought/Y.npy')
EVI = np.load('/data/NDVI/MOD13Q1_EVI3D.npy')
ws = 11
o = 6
#EVIsg = np.load('/data/emily/WF/Drought/EVIsg.npy')

########################################################################
#3D matrix masking
#########################################################################
#find double-cropped pixels using PC3
PC3 = Y[:,2].reshape((1, 1927,1082))
PC3m = np.repeat(PC3, 253, axis = 0)
loading = 3
EVIm = np.ma.masked_where(PC3m < loading, EVI)
nanvalue = -9999.0
EVInan = EVIm.filled([nanvalue]) #3d datacube 253, 1927, 1082


########################################################################
#2D matrix masking
#########################################################################
#reshape with rows as locations and columns as time
EVI2D = np.transpose(EVIm.reshape((EVIm.shape[0], EVIm.shape[1]*EVIm.shape[2])))
EVI2Dnan = EVI2D.filled(nanvalue) #reshape 2d 2085014 x 253
index = np.where(EVI2Dnan!=nanvalue) #index in 2d matrix where data not masked
data = []
for row in range(0,EVI2D.shape[0]):
	if EVI2Dnan[row,1] != nanvalue:
		data.append(EVI2Dnan[row,:])
EVIdc = np.asarray(data) #data subset with dimensions 34249 x 253
#check EVI dc19854 == np.ma.count(EVIm[1,:,:])
#first row here is the first indexed value in the 2D array.  Replace by indices and reshape to get 3D

	
########################################################################
#adaptation in Yala 2014: MAX VALUE
########################################################################
y2014 = EVI2D[:,226:253]  #226 is October 2013
y2014nan = y2014.filled(nanvalue)
index_yala = np.where(y2014nan[:,1]!=nanvalue)
index_yala_a = np.asarray(index_yala)
data = []
for row in range(0, y2014.shape[0]):
	if y2014nan[row,1] != nanvalue:
		data.append(y2014nan[row,:])

EVIdcyala = np.asarray(data)  
yala = EVIdcyala[:,0:10]  #changed to MAHA, October 2013 to end of March 2014

yalasg = []
maxrow = []
for r in range(0,yala.shape[0]):
	pixel = yala[r,:]
	sg = savitzky_golay(pixel, ws, o)
	maxr  = max(sg)
	maxrow.append(maxr)
	yalasg.append(sg)
yala_sg = np.asarray(yalasg)
max_row = np.asarray(maxrow)

#find time series of loadings
index_yalamax = np.where(max_row > 0.6)
out = np.asarray(index_yalamax)
index_yalamax_a = out.reshape((out.shape[1]))
yalats = yala[index_yalamax_a,:]   #time series of cropped pixels!!!!

#plotting back in original shape
#index_yalamax indexes 34... array > 0.5
nonmasked = np.zeros((EVIdc.shape[0]))
full = np.zeros((y2014.shape[0]))

nonmasked[index_yalamax_a] = 1
#index _yala_a non masked rows in 20...
full[index_yala_a] = nonmasked
ad_mask = full.reshape((1927, 1082))  #adaptive pixels with max method




########################################################################
#adaptive pixels with integral
########################################################################

intdata = yala_sg #smoothed dataset for 2014 yala, 10 observations from 8:18

intval = []

for r in range(intdata.shape[0]):
	fxn = intdata[r,:]
	out = integrate.simps(fxn,dx=1, axis=0)
	intval.append(out)

intval = np.asarray(intval)

index_int = np.where(intval>(intval.mean() + intval.std()))
index_int_a = np.asarray(index_int)
index_int_a = index_int_a.reshape((index_int_a.shape[1]))
yalaint = yala[index_int_a,:] 

#plotting back in original shape
#index_yalamax indexes 34... array > 0.5
nonmasked = np.zeros((EVIdc.shape[0]))
full = np.zeros((y2014.shape[0]))

nonmasked[index_int_a] = 1
#index _yala_a non masked rows in 20...
full[index_yala_a] = nonmasked
int_mask = full.reshape((1927, 1082))  #adaptive pixels with max method



#find combo criteria
combo_06sd = int_mask + ad_mask
combo_05mean = int_mask + ad_mask



#gis database
GISfull = np.load('/media/data/NDVI/columns/GIS.npy')
GIS = GISfull[0:1927*1082, :]
ae = GIS[:,0].reshape((1927,1082))
casc = GIS[:,1].reshape((1927,1082))
idpt = GIS[:,4].reshape((1927,1082))
lu = GIS[:,5].reshape((1927,1082))
masl = GIS[:,6].reshape((1927,1082))
maslsys = GIS[:,7].reshape((1927,1082))
rd = GIS[:,9].reshape((1927,1082))
notnk = GIS[:,8].reshape((1927,1082))
satnk = GIS[:,10].reshape((1927,1082))
sl = GIS[:,11].reshape((1927,1082))
test = lu[combo_05mean==2]
GWP = np.load('/media/data/NDVI/columns/GWP.npy').reshape((253, 1927, 1082))
pop14 = GWP[252,:,:].reshape((1927, 1082))



















PCA14 = PCA(EVIdc)
#weights
Wt14 = PCA14.Wt
#rows are the PC axes
#my data in terms of the principal component axes
Y14 = PCA14.Y
ve = PCA14.fracs

#find time series of loadings
PC1 = Y14[:,0]
index = np.where(PC1>3)
index_a = np.asarray(index)
ts = EVIdc[index_a].reshape((index_a.shape[1], 23))

#plotting back in original chape
index_arr = np.asarray(index).reshape(34249)
full = y2014
full[index_arr] = Y14
#new full shows PCs

PC = full[:,0].reshape((1927, 1082))





	
	
	
#KNN and DTW
#DTW: http://nbviewer.ipython.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb
#http://nbviewer.ipython.org/github/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping/blob/master/K_Nearest_Neighbor_Dynamic_Time_Warping.ipynb





#plotting
fig, ax = plt.subplots()
fig.canvas.draw()
year = np.arange(0,253,23)
plt.xticks(year, fontsize=25)
plt.yticks(fontsize = 25)
plt.grid(True)
labels = np.arange(2004, 2015, 1)
ax.set_xticklabels(labels)
#labels = ['A', 'M', 'M', 'Jn', 'Jn', 'Jy', 'Jy', 'Au', 'Au', 'S', 'S']

#plotting adaptive pixels
idx = np.asarray(np.where(PC2>3))


#all yalas
start = np.arange(8,253, 23) #11 so I'm sure I don't get Maha residue
yalas = []
for s in range(start.shape[0]):
	data = EVIdc[:,s:s+10]
	yalas.append(data)
yalas = np.asarray(yalas)

#max for each smoothed yala
hist = []
sg = []
ws = 11
o = 6
sgfull = []
maxfull = []

for d in range(yalas.shape[0]):
	sg = []
	maxrow = []
	for r in range(yalas.shape[1]):
		sgr = savitzky_golay(yalas[d,r,:], ws, o)
		maxr = max(sgr)
		maxrow.append(maxr)
		sg.append(sgr)
	sg = np.asarray(sg)
	mr = np.asarray(maxrow)
	sgfull.append(sg)
	maxfull.append(mr)

sgar = np.asarray(sgfull)
maxarr = np.asarray(maxfull)

#plotting
plt.subplots(10,1)
for i in range(10):
	plt.subplot(10,1,i)
	plt.hist(maxarr[i,:], 200)
	plt.xlim([0.3,0.9])
	plt.xticks([])
	plt.yticks([])
	plt.axvline(maxarr[i,:].mean(), color = 'r', linewidth=2.0)


