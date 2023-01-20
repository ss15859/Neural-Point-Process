import pandas as pd 
import numpy as np
import csv
import os
import math
import sys
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from labellines import labelLine, labelLines

from skmisc.loess import loess

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import ETAS
from utils import truncate_catalog_by_threshold, datetime_to_hours,maxc
from NeuralPP import NPP
from AVN_experiment import select_training_testing_partition

sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2.2)

##### Input variables of script
earthquake_for_partition = sys.argv[1]
M0pred = 3
time_step = 20

HighestMcut = 3.0
if earthquake_for_partition == 'Campotosto':
	lowestMcut = 1.4
else:
	lowestMcut = 1.2


#### reading catalog

AVN_catalog = pd.read_csv('../data/Catalogs/Amatrice_CAT5.v20210504_reduced_cols.csv')

AVN_catalog = datetime_to_hours(AVN_catalog)

AVN_catalog = AVN_catalog.dropna()


##### subsetting catalog

truncated_catalog = truncate_catalog_by_threshold(AVN_catalog,M0pred)


times = np.array(truncated_catalog['time'])
mags = np.array(truncated_catalog['mw'])


timeupto = select_training_testing_partition(earthquake_for_partition)

T_train = times[times<timeupto]
M_train = mags[times<timeupto]
T_test = times[times>=timeupto]
M_test = mags[times>=timeupto]


############## reading results

n= round((HighestMcut-lowestMcut)*10+1)

vec = [None]*n; nppcollated_LL = [None]*n; npp_mag = [None]*n; ETAS_mag = [None]*n

for idx, Mcut in enumerate(np.arange(lowestMcut,HighestMcut+0.2,0.2)):

	Mcut = round(Mcut,1)
	filename  = '../data/Results/resultsMcut-'+str(Mcut)+'_partition:' + str(earthquake_for_partition)+'_M0pred:'+str(M0pred)+'.csv'
	D = pd.read_csv(filename)
	vec[idx] = np.array(D.ETAS_pointwise_like)
	nppcollated_LL[idx] = np.array(D.NN_pointwise_lik)
	npp_mag[idx]   = np.array(D.nppLLmag)
	ETAS_mag[idx] = np.array(D.ETASLLmarkvec)

################################################################################ CIG (times)

fig3,axs3twin = plt.subplots(1)

axs3 = axs3twin.twiny()

cmap = matplotlib.cm.get_cmap('viridis')

for idx,Mcut in enumerate(np.arange(lowestMcut,HighestMcut+0.2,0.2)):

	col = idx/10
	Mcut = np.round(Mcut,1)
	axs3.plot(np.cumsum(np.ones_like(T_test[:len(vec[idx])])),np.cumsum((nppcollated_LL[idx]-vec[idx])),label=str(Mcut),linewidth =2.5,color = cmap(col))
	axs3twin.plot(np.cumsum(np.ones_like(T_test[:len(vec[idx])])),np.cumsum((nppcollated_LL[idx]-vec[idx])),label=str(Mcut),linewidth =2.5,color = cmap(col))# Create a dummy plot

axs3.legend(bbox_to_anchor=(1.11,0.95),prop={'size': 18},title='Mcut')

labelLines(axs3.get_lines(),align=False, fontsize=18)
axs3.axvline(9,linestyle='--',color='#cb6952ff',zorder=1)

axs3.axvline(sum(T_test<3755),linestyle='--',color='#cb6952ff',zorder=0)
axs3.axhline(0,linestyle='dotted',color='grey')



def transform_time(number):
	return str(int(round(T_test[number]-T_test[9],0)))   # hours since Norcia

positions = [9,200,300,400,500,600,700,800]
string_positions = [str(position) for position in positions]


labels = [transform_time(number) for number in positions]

axs3.set_xlabel('Hours Since Norcia',labelpad=12.0)

axs3twin.set_ylabel('  CIG \n time',rotation=0,labelpad=35.0)

axs3.set_xticks(positions) 
axs3twin.set_xticks(positions)
axs3.set_xticklabels(labels)
axs3twin.set_xticklabels(string_positions)


plt.setp(axs3twin.get_xticklabels(), visible=False)

axs3twin.tick_params(axis='x', which='both', length=0)


axs3.annotate('Norcia', (9,1350), xytext=(40, 1320), 
    bbox=dict(boxstyle="round", alpha=0.1), 
    arrowprops = dict(arrowstyle="simple",color = '#be2a35'))
axs3.annotate('Campotosto',(sum(T_test<3755),1350),zorder=10000, xytext=(546, 1320), 
    bbox=dict(boxstyle="round", alpha=0.1,zorder=1000000), 
    arrowprops = dict(arrowstyle="simple",color = '#be2a35',zorder=1000000))

########################################################################################### CIG (magnitudes)

fig4,axs4twin = plt.subplots(1)

axs4 = axs4twin.twiny()

for idx,Mcut in enumerate(np.arange(lowestMcut,HighestMcut+0.2,0.2)):
	col = idx/10
	Mcut = np.round(Mcut,1)
	axs4.plot(np.cumsum(np.ones_like(T_test[:len(ETAS_mag[idx])])),np.cumsum((npp_mag[idx]-ETAS_mag[idx])),label=str(Mcut),linewidth =2.5,color = cmap(col))
	axs4twin.plot(np.cumsum(np.ones_like(T_test[:len(ETAS_mag[idx])])),np.cumsum((npp_mag[idx]-ETAS_mag[idx])),label=str(Mcut),linewidth =2.5,color = cmap(col))# Create a dummy plot


labelLines(axs4.get_lines(),align=False, fontsize=18)
axs4.axvline(9,linestyle='--',color='#cb6952ff',zorder=1)
axs4.axvline(sum(T_test<3755),linestyle='--',color='#cb6952ff',zorder=0)
axs4.axhline(0,linestyle='dotted',color='grey')


axs4twin.set_xlabel('Earthquake Number')
axs4twin.set_ylabel('  CIG \n mag',rotation=0,labelpad=35.0)

plt.setp(axs4.get_xticklabels(), visible=False)
plt.setp(axs4.get_yticklabels(), visible=False)
axs4.tick_params(axis='both', which='both', length=0)

axs4.set_xticks(positions) 
axs4twin.set_xticks(positions)
axs4.set_xticklabels(labels)
axs4twin.set_xticklabels(string_positions)


axins = zoomed_inset_axes(axs4, 3,loc=3,bbox_to_anchor=[420,200]) 

for idx,Mcut in enumerate(np.arange(lowestMcut,HighestMcut+0.2,0.2)):
	col = idx/10
	Mcut = np.round(Mcut,1)
	axins.plot(np.cumsum(np.ones_like(T_test[:len(ETAS_mag[idx])])),np.cumsum((npp_mag[idx]-ETAS_mag[idx])),label=str(Mcut),linewidth =2.5,color = cmap(col))
	axins.axhline(0,linestyle='dotted',color='grey')
	axins.axvline(9,linestyle='--',color='#cb6952ff',zorder=1)


# sub region of the original image
x1, x2, y1, y2 = -1, 130, -15, 15
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

axins.spines["bottom"].set_edgecolor('black')  
axins.spines["bottom"].set_linewidth(2)
axins.spines["left"].set_edgecolor('black')  
axins.spines["left"].set_linewidth(2)
axins.spines["right"].set_edgecolor('black')  
axins.spines["right"].set_linewidth(2)
axins.spines["top"].set_edgecolor('black')  
axins.spines["top"].set_linewidth(2)

mark_inset(axs4, axins, loc1=2, loc2=4 ,fc="none", ec="black",zorder=1000)

plt.show()


########### Gain against completeness plot

plot_catalog = truncate_catalog_by_threshold(AVN_catalog,0)

comp_T = plot_catalog.time[plot_catalog.time>timeupto]
comp_M = plot_catalog.mw[plot_catalog.time>timeupto]

## estimate completeness over time

window_size=300
nwindows = math.floor(len(comp_M)/window_size)

Mc_t = np.zeros(nwindows)
mid_time = np.zeros(nwindows)

for i in range(nwindows):
    
    mid_time[i] = np.mean(comp_T[i*window_size:(i+1)*window_size])
    window = comp_M[i*window_size:(i+1)*window_size]
    Mc_t[i] = maxc(window,0.05)

hours = mid_time[mid_time!=0]
mgs = Mc_t[mid_time!=0]


Mc = np.zeros_like(T_test)

for i, value in enumerate(T_test):
    idx = (np.abs(hours - value)).argmin()

    Mc[i] = mgs[idx]


#### lowess regression of IG against completeness

x = Mc
y = nppcollated_LL[0]-vec[0]

y = [x for _,x in sorted(zip(x,y))]
x = sorted(x)

l = loess(x,y)
l.fit()
pred = l.predict(x, stderror=True)
conf = pred.confidence()

lowess = pred.values
ll = conf.lower
ul = conf.upper

plt.plot(x, lowess,color='#61a889ff')
plt.fill_between(x,ll,ul,alpha=.33,color='#61a889ff')
plt.ylim([0,None])
plt.xlabel(r'$M_0(t)$')
plt.ylabel('Log-Likelihood gain')
plt.legend(title='Mcut = 1.2',fontsize=13)

plt.show()