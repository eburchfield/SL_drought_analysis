import numpy as np
import os
import pylab as plt

d = '/home/emily/Desktop/MODIS/WF/Drought'
filename = '/media/data/NDVI/columns/MOD13Q1_EVI.npy'
bands = 253
rows = 1927
cols = 1082

ds = np.load(filename)
data = ds.reshape((bands, rows, cols))

EVI = np.ma.masked_where(data == data[1,:,:].min(), data)
EVIfull = np.load('/media/data/NDVI/MOD13Q1_EVI3D.npy')

month = np.arange(0,12,1)
index = np.arange(0, 230, 23) #don't include 2014
year = 230 #start date for year of interest

#quality mask
A = []
average = []
#anomaly
for m in month:
	month_avg = []
	for i in index:
		month_avg.append(EVI[i,:,:])
	month_avg_arr = np.asarray(month_avg)
	month_ma = np.ma.masked_where(month_avg_arr == month_avg_arr.min(), month_avg_arr)
	avg = np.mean(month_ma, axis = 0)
	a = EVI[year+m,:,:] - avg
	A.append(a)
	average.append(avg)

EVIA= np.asarray(A)
EVIA = np.ma.masked_where(EVIA == EVIA.min(), EVIA)
average = np.asarray(average)
avgm = np.ma.masked_where(average==average[1,1,1], average)
out = avgm.filled(nanvalue)


#no quality mask

A = []
#anomaly
for m in month:
	month_avg = []
	for i in index:
		month_avg.append(EVIfull[i,:,:])
	month_avg_arr = np.asarray(month_avg)
	#month_ma = np.ma.masked_where(month_avg_arr == month_avg_arr.min(), month_avg_arr)
	avg = np.mean(month_avg_arr, axis = 0)
	a = EVIfull[year+m,:,:] - avg
	A.append(a)

EVInoq= np.asarray(A)
EVInoq = np.ma.masked_where(EVInoq == EVInoq[1,1,1], EVInoq)


#make quality movie
os.mkdir(str(d) + '/EVIAnoQA')

cmap = plt.cm.PiYG
cmap.set_bad('k') #set no data to white

month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'] 

for b in range(0, EVIA.shape[0]):
    fig = plt.figure(figsize=(7,7))
    plt.imshow(EVIA[b,:,:], cmap=cmap, vmin=-0.9, vmax=0.9)     #change array
    file_id = str(b+1).zfill(3) #001
    plt.title(str(month[b]))
    plt.colorbar()
    plt.savefig(str(d) + '/EVIA_film/' + str(file_id) + '.jpg')            #change directory
    plt.close(fig)
    
#make movie 
cmd = 'convert -delay 200 -loop 0 /home/emily/Desktop/MODIS/WF/Drought/EVIA_film/*.jpg /home/emily/Desktop/MODIS/WF/Drought/EVIA_film/EVIA.gif'
os.system(cmd)

#plot single image
fig = plt.figure(figsize = (8,8))
plt.imshow(avg, cmap=cmap, vmin =0, vmax=1)
plt.colorbar()
