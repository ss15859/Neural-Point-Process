import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
import datetime
sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2.2)

from utils import maxc


#################################################################### Reading Data

Amatrice = pd.read_csv('../data/Catalogs/Amatrice_CAT5.v20210504_reduced_cols.csv')

##### converting datetime to hours

Amatrice['datetime'] = pd.to_datetime(Amatrice[['year', 'month', 'day', 'hour', 'minute','second']])
Amatrice['time'] = (Amatrice['datetime']-Amatrice['datetime'][0])/ pd.to_timedelta(1, unit='H')

##### removing excess columns and missing data

Amatrice = Amatrice[['time','mw','datetime']]
Amatrice = Amatrice.dropna()


comp_T = Amatrice.time
comp_M = Amatrice.mw


############## estimating the completeness

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

##################


date = Amatrice['datetime'][0] + pd.to_timedelta(hours, unit='H')


plt.step(date[2:],mgs[2:],color = '#be2a35',label =r'$M_0(t)$',where='post')


plot_catalog = Amatrice[Amatrice.time<4800]

z = (10**plot_catalog['mw'])/1000  

sortcat = np.sort(np.array(plot_catalog['mw']))

Am = np.where(np.array(plot_catalog['mw'])==sortcat[-1])
No = np.where(np.array(plot_catalog['mw'])==sortcat[-2])
Vi = np.where(np.array(plot_catalog['mw'])==sortcat[-4])[0]
Ca = np.where(np.array(plot_catalog['mw'])==sortcat[-7])


xx = np.array(plot_catalog['datetime'])
yy = 6.7


plt.scatter(np.array(plot_catalog['datetime']),plot_catalog['mw'],s=z,color='#36423cff')
plt.annotate('Amatrice', (xx[Am],6.7), xytext=(xx[Am], yy+0.5), 
    bbox=dict(boxstyle="round", alpha=0.1), 
    arrowprops = dict(arrowstyle="simple",color = '#be2a35'))
plt.annotate('Norcia', (xx[No],6.7), xytext=(xx[No], yy+0.5), 
    bbox=dict(boxstyle="round", alpha=0.1), 
    arrowprops = dict(arrowstyle="simple",color = '#be2a35'))
plt.annotate('Visso', (xx[Vi],6.7), xytext=(xx[Vi-30000], yy+0.5), 
    bbox=dict(boxstyle="round", alpha=0.1), 
    arrowprops = dict(arrowstyle="simple",color = '#be2a35'))
plt.annotate('Campotosto', (xx[Ca],6.7), xytext=(xx[Ca], yy+0.5), 
    bbox=dict(boxstyle="round", alpha=0.1), 
    arrowprops = dict(arrowstyle="simple",color = '#be2a35'))
plt.vlines(xx[Am],ymin=-0.5,ymax=7,linestyle='--',color='#cb6952ff',zorder=0)
plt.vlines(xx[No],ymin=-0.5,ymax=7,linestyle='--',color='#cb6952ff',zorder=0)
plt.vlines(xx[Vi],ymin=-0.5,ymax=7,linestyle='--',color='#cb6952ff',zorder=0)
plt.vlines(xx[Ca],ymin=-0.5,ymax=7,linestyle='--',color='#cb6952ff',zorder=0)
plt.ylabel('Mw')
plt.xlabel('Date')
plt.ylim([-0.5,7])
plt.legend()

plt.show()
