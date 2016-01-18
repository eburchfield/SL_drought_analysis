#extract the time series of the adaptive pixels in particular seads communities

import numpy as np
import os

os.chdir('/home/emily/Desktop/MODIS/WF/Drought')


evi = np.load('/media/data/NDVI/columns/MOD13Q1_EVI.npy').reshape((253, 1927, 1082))
Y = np.load('Y.npy')
gn = np.load('/home/emily/Desktop/MODIS/WF/SEADS/seads_gns.npy')
gnt = np.tile(gn, (253, 1,1))
pc = Y[:,2].reshape((1927, 1082))
pct = np.tile(pc, (253, 1,1))
ap5 = np.load('combo_05mean.npy')
ap5t = np.tile(ap5, (253, 1,1))

evi_5m = np.ma.masked_where(ap5t<1, evi)  #adaptive pixel mask
nanvalue = -9999.0
evi_5m = np.ma.filled(evi_5m, nanvalue)


idx = np.where(gnt==i)
idxa = np.asarray(idx)

data = evi_5m[idx]
t1 = data.reshape((data.shape[0]/253, 253))
 





gnts = []	
i=1	
evign = np.ma.masked_where(gnt!=i, evi_5m)
evign = np.ma.filled(evign, nanvalue)
data = []
for r in range(1927):
	for c in range(1082):
		if evign[1,r,c]!=nanvalue:
			ts = evign[:,r,c].reshape((253,1))
			data.append(ts)
gnts.append(data)
				
