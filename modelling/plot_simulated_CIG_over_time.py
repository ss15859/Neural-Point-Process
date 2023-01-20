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

import ETAS

from utils import maxc

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

sns.set_theme()

sns.set_context("paper")
sns.set(font_scale=2.2)

##### Input variables of script
type_of_data = sys.argv[1]

M0pred = 3.0
time_step = 20
timeupto= 4000

HighestMcut = 3.0
if type_of_data == 'incomplete':
	lowestMcut = 1.2
	incomplete = True
else:
	lowestMcut = 1.8
	incomplete = False


##### reading 

synthetic = pd.read_csv('../data/Catalogs/synthetic_' + type_of_data + '_catalog.csv')
synthetic['time'] = pd.to_datetime(synthetic['time'])
synthetic = synthetic.sort_values('time')

times = ETAS.datetime64_to_days(synthetic['time'])
mags = np.array(synthetic['magnitude'])

##### subsetting catalog

T_test = times[times>=timeupto]
M_test = mags[times>=timeupto]

times_of_targets = T_test[M_test>=M0pred]
M_test = M_test[M_test>=M0pred]


### reading results

n= round((HighestMcut-lowestMcut)*10+1)

vec = [None]*n; nppcollated_LL = [None]*n; npp_mag = [None]*n; ETAS_mag = [None]*n

for idx, Mcut in enumerate(np.arange(lowestMcut,HighestMcut+0.2,0.2)):

	Mcut = round(Mcut,1)
	filename  = '../data/Results/'+type_of_data +'_simulated_resultsMcut-'+str(Mcut)+'_M0pred:'+ str(M0pred)+ '.csv'
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
	axs3.plot(np.cumsum(np.ones_like(times_of_targets[:len(vec[idx])])),np.cumsum((nppcollated_LL[idx]-vec[idx])),label=str(Mcut),linewidth =2.5,color = cmap(col))
	axs3twin.plot(np.cumsum(np.ones_like(times_of_targets[:len(vec[idx])])),np.cumsum((nppcollated_LL[idx]-vec[idx])),label=str(Mcut),linewidth =2.5,color = cmap(col))# Create a dummy plot

	
axs3.legend(bbox_to_anchor=(1.11,0.95),prop={'size': 18},title='Mcut')

labelLines(axs3.get_lines(),align=False, fontsize=18)
if incomplete:
	axs3.axvline(96,linestyle='--',color='#cb6952ff',zorder=1)
	axs3.axvline(156,linestyle='--',color='#cb6952ff',zorder=0)

axs3.axhline(0,linestyle='dotted',color='grey')


def transform_time(number):
	return str(int(round(times_of_targets[number]-times_of_targets[96],0)))

positions = [0,100,200,300,400,500]
if not incomplete:
	positions.append(600)
string_positions = [str(position) for position in positions]


labels = [transform_time(number) for number in positions]


if incomplete:
	axs3.set_xlabel('Hours since start of incompleteness',labelpad=12.0)
else:
	axs3.set_xlabel('Earthquake Number',labelpad=12.0)

axs3twin.set_ylabel('  CIG \n time',rotation=0,labelpad=35.0)

if incomplete:
	axs3.set_xticks(positions) 
	axs3twin.set_xticks(positions)

	axs3.set_xticklabels(labels)
	axs3twin.set_xticklabels(string_positions)


plt.setp(axs3twin.get_xticklabels(), visible=False)

axs3twin.tick_params(axis='x', which='both', length=0)


axs3.annotate('Norcia', (9,1350), xytext=(40, 1320), 
    bbox=dict(boxstyle="round", alpha=0.1), 
    arrowprops = dict(arrowstyle="simple",color = '#be2a35'))
axs3.annotate('Campotosto',(sum(times_of_targets<3755),1350),zorder=10000, xytext=(546, 1320), 
    bbox=dict(boxstyle="round", alpha=0.1,zorder=1000000), 
    arrowprops = dict(arrowstyle="simple",color = '#be2a35',zorder=1000000))


########################################################################################### CIG (magnitudes)

fig4,axs4twin = plt.subplots(1)

axs4 = axs4twin.twiny()

for idx,Mcut in enumerate(np.arange(lowestMcut,HighestMcut+0.2,0.2)):

	col = idx/10
	Mcut = np.round(Mcut,1)
	axs4.plot(np.cumsum(np.ones_like(times_of_targets[:len(ETAS_mag[idx])])),np.cumsum((npp_mag[idx]-ETAS_mag[idx])),label=str(Mcut),linewidth =2.5,color = cmap(col))
	axs4twin.plot(np.cumsum(np.ones_like(times_of_targets[:len(ETAS_mag[idx])])),np.cumsum((npp_mag[idx]-ETAS_mag[idx])),label=str(Mcut),linewidth =2.5,color = cmap(col))# Create a dummy plot


labelLines(axs4.get_lines(),align=False, fontsize=18)
if incomplete:
	axs4.axvline(96,linestyle='--',color='#cb6952ff',zorder=1)
	axs4.axvline(156,linestyle='--',color='#cb6952ff',zorder=0)

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


plt.show()



############## Gain against completeness plot

if type_of_data == 'incomplete':

	comp_T_temp = times[times>timeupto]
	comp_T = comp_T_temp
	comp_M = mags[times>timeupto]


	chopped_times=T_test[T_test>timeupto]
	times_of_targets = chopped_times[M_test[T_test>timeupto]>=M0pred]

	window_size=100
	nwindows = math.floor(len(comp_M)/window_size)

	Mc_t = np.zeros(nwindows)
	mid_time = np.zeros(nwindows)

	for i in range(nwindows):
	    
	    mid_time[i] = np.mean(comp_T[i*window_size:(i+1)*window_size])
	    window = comp_M[i*window_size:(i+1)*window_size]
	    Mc_t[i] = maxc(window,0.05)

	hours = mid_time[mid_time!=0]
	mgs = Mc_t[mid_time!=0]


	# ## asign a value of completeness for each event

	Mc = np.zeros_like(times_of_targets)

	for i, value in enumerate(times_of_targets):
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
	plt.hlines(0,min(x),max(x),linestyle='--',color='#cb6952ff')
	plt.xlabel(r'$M_0(t)$')
	plt.ylabel('Log-Likelihood gain')
	plt.legend(title='Mcut = 1.2',fontsize=13)
	plt.show()

