import pandas as pd 
import numpy as np
import csv
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from mpltools import annotation
import ETAS
sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2.2)


type_of_data = str(sys.argv[1])

if type_of_data == 'complete':
    lowestMcut = 1.7
elif type_of_data == 'incomplete':
    lowestMcut = 1.1

M0pred = 3.0
HighestMcut = 3.0

n= round((HighestMcut-lowestMcut)*10+1)
train_Mcuts = np.linspace(lowestMcut,HighestMcut,n)

######## Read in results

ETAStemp = np.zeros(n)
NNtemp = np.zeros(n)
ETASmark = np.zeros(n)
NNmark = np.zeros(n)


LLpoiss = np.zeros(n)

mus = np.zeros(n)

NN_sub_mean = np.zeros(n) 
ETAS_sub_mean = np.zeros(n)
Poiss_sub_mean = np.zeros(n)

ntrain  = np.zeros(n)

NNtime = np.zeros(n)
ETAStime = np.zeros(n)

ETAS_mu = np.zeros(n)
ETAS_k0 = np.zeros(n)
ETAS_a = np.zeros(n)
ETAS_c = np.zeros(n)
ETAS_omega = np.zeros(n)

row=[]

magnitude_gain = []

for i, Mcut in enumerate(train_Mcuts):

	Mcut = round(Mcut,1)

	filename  = '../data/Results/'+ type_of_data + '_simulated_resultsMcut-'+str(Mcut)+'_M0pred:'+str(M0pred)+'.csv'
	D = pd.read_csv(filename)

	LLpoiss[i] = D.LLpoiss[0]
	ntrain[i] = D.ntrain[0]
	NNtime[i] = 8*pd.to_timedelta(D.NN_train_time[0])/ np.timedelta64(1, 'h')
	ETAStime[i] = pd.to_timedelta(D.ETAS_train_time[0])/ np.timedelta64(1, 'h')

	sub_vec = np.array(D.ETAS_pointwise_like)
	sub_npp = np.array(D.NN_pointwise_lik)
	sub_vec -= LLpoiss[i]
	sub_npp -= LLpoiss[i]

	npp_mag   = np.array(D.nppLLmag)
	ETAS_mag = np.array(D.ETASLLmarkvec)


	ETAS_mu[i] = D.ETAS_mu[0]
	ETAS_k0[i] = D.ETAS_k0[0]
	ETAS_a[i] = D.ETAS_a[0]
	ETAS_c[i] = D.ETAS_c[0]
	ETAS_omega[i] = D.ETAS_omega[0]

	for j in range(len(sub_vec)):

		row.append([sub_vec[j],'ETAS',round(train_Mcuts[i],1), ntrain[i]])
		row.append([sub_npp[j],'NN',round(train_Mcuts[i],1),ntrain[i]])

		magnitude_gain.append([npp_mag[j],'NN',round(train_Mcuts[i],1), ntrain[i]])
		magnitude_gain.append([ETAS_mag[j],'ETAS',round(train_Mcuts[i],1), ntrain[i]])


ntest = len(sub_vec)


########### Likelihood plots


subset_mag = pd.DataFrame(row, columns=["Log-likelihood Score", "model","Mcut","N train"])
my_dpi=96
fig, ax1 = plt.subplots(1,2,figsize=(1914/my_dpi, 663/my_dpi), dpi=my_dpi)

fig.subplots_adjust(top=0.953,bottom=0.165,left=0.063,right=0.911,hspace=0.205,wspace=0.206)
sns.barplot(data = subset_mag, x="Mcut", y="N train", alpha=0.6,color='#61a889ff', ax=ax1[0])

for i,t in enumerate(ax1[0].get_xticklabels()):
	tt = int(10*float(t.get_text()))
	if (tt % 2) != 0:
		t.set_visible(False)

ax2 = ax1[0].twinx()
start, end, stepsize = ax1[0].get_yticks()[0], ax1[0].get_yticks()[-1], (ax1[0].get_yticks()[-1]-ax1[0].get_yticks()[0])/(len(ax1[0].get_yticks())-1)

ax1[0].yaxis.set_ticks(np.arange(0, 30000, 2500))
ax1[0].yaxis.tick_right()
ax1[0].yaxis.set_label_position("right")
ax1[0].set_ylabel('', color = '#417961ff')
ax1[0].tick_params(axis='y', colors='#417961ff',labelright=False,right=False)


ax1[0].grid(False)
sns.pointplot(x="Mcut", y="Log-likelihood Score", data=subset_mag,hue="model",palette=['#36423cff','#cb6952ff'],ax=ax2)
ax2.yaxis.tick_left()

ax2.yaxis.set_label_position("left")
ax2.set_ylabel('Log-likelihood Score',labelpad=-6.0)

ax2.legend(title='N test = ' + str(ntest) ,fontsize=18, title_fontsize=18 )

Dataframe_mag = pd.DataFrame(magnitude_gain, columns=["Log-likelihood Score", "model","Mcut","N train"])


sns.barplot(data = Dataframe_mag, x="Mcut", y="N train", alpha=0.6,color='#61a889ff', ax=ax1[1])
for i,t in enumerate(ax1[1].get_xticklabels()):
	tt = int(10*float(t.get_text()))
	if (tt % 2) != 0:
		t.set_visible(False)
ax4 = ax1[1].twinx()
start, end, stepsize = ax1[1].get_yticks()[0], ax1[1].get_yticks()[-1], (ax1[1].get_yticks()[-1]-ax1[1].get_yticks()[0])/(len(ax1[1].get_yticks())-1)

ax1[1].yaxis.set_ticks(np.arange(0, 30000, 2500))
ax1[1].yaxis.tick_right()
ax1[1].yaxis.set_label_position("right")
ax1[1].set_ylabel('Training size', color = '#417961ff')
ax1[1].tick_params(axis='y', colors='#417961ff')
ax1[1].grid(False)
sns.pointplot(x="Mcut", y="Log-likelihood Score", data=Dataframe_mag,hue="model",palette=['#cb6952ff','#36423cff'],ax=ax4)
ax4.yaxis.tick_left()
ax4.yaxis.set_label_position("left")
ax4.set_ylabel('').set_visible(False)
ax4.legend().set_visible(False)
ax1[1].legend().set_visible(False)



#### train time plot


if type_of_data == 'complete':

	time_fig, time_ax = plt.subplots()

	time_ax.plot(ntrain,NNtime, '-o',label='NN',color= '#cb6952ff')
	time_ax.plot(ntrain,ETAStime, '-o',label = 'ETAS',color = '#36423cff')

	time_ax.set_xlabel('Training events')
	time_ax.set_ylabel('CPU Hours to converge to MLE')

	ymin, ymax = time_ax.get_ylim()

	time_ax.set_yscale('log')
	time_ax.set_xscale('log')
	time_ax.legend()
	annotation.slope_marker((7000, 10), 2, ax=time_ax,
		text_kwargs={'color': '#61a889ff'},
	                        poly_kwargs={'facecolor': '#61a889ff'})
	annotation.slope_marker((7000, 0.3), 1, ax=time_ax,
		text_kwargs={'color': '#61a889ff'},
	                        poly_kwargs={'facecolor': '#61a889ff'})


plt.show()

