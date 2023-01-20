import pandas as pd 
import numpy as np
import csv
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2.2)


earthquake_for_partition = sys.argv[1]
M0pred = 3
HighestMcut = 3.0
if earthquake_for_partition == 'Campotosto':
	lowestMcut = 1.3
else:
	lowestMcut = 1.2

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

ETAS_mu = np.zeros(n)
ETAS_k0 = np.zeros(n)
ETAS_a = np.zeros(n)
ETAS_c = np.zeros(n)
ETAS_omega = np.zeros(n)



row=[]
magnitude_gain = []


for i, Mcut in enumerate(train_Mcuts):

	Mcut = round(Mcut,1)

	filename  = '../data/Results/resultsMcut-'+str(Mcut)+'_partition:' + str(earthquake_for_partition)+'_M0pred:'+str(M0pred)+'.csv'
	D = pd.read_csv(filename)

	LLpoiss[i] = D.LLpoiss[0]
	ntrain[i] = D.ntrain[0]

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

ax1[0].yaxis.set_ticks(np.arange(0,140000,20000))
ax1[0].yaxis.tick_right()
ax1[0].yaxis.set_label_position("right")
ax1[0].set_ylabel('', color = '#417961ff')
ax1[0].tick_params(axis='y', colors='#417961ff',labelright=False,right=False)

ax1[0].grid(False)
sns.pointplot(x="Mcut", y="Log-likelihood Score", data=subset_mag,hue="model",palette=['#36423cff','#cb6952ff'],ax=ax2)
ax2.yaxis.tick_left()
ax2.set_ylim([0.8,3.9])
ax2.yaxis.set_label_position("left")
ax2.set_ylabel('Log-likelihood Score')

ax2.legend(title='N test = ' + str(ntest),fontsize=18, title_fontsize=18)

Dataframe_mag = pd.DataFrame(magnitude_gain, columns=["Log-likelihood Score", "model","Mcut","N train"])


sns.barplot(data = Dataframe_mag, x="Mcut", y="N train", alpha=0.6,color='#61a889ff', ax=ax1[1])
for i,t in enumerate(ax1[1].get_xticklabels()):
	tt = int(10*float(t.get_text()))
	if (tt % 2) != 0:
		t.set_visible(False)

ax4 = ax1[1].twinx()
start, end, stepsize = ax1[1].get_yticks()[0], ax1[1].get_yticks()[-1], (ax1[1].get_yticks()[-1]-ax1[1].get_yticks()[0])/(len(ax1[1].get_yticks())-1)

ax1[1].yaxis.set_ticks(np.arange(0,140000,20000))
ax1[1].yaxis.tick_right()
ax1[1].yaxis.set_label_position("right")
ax1[1].set_ylabel('Training size', color = '#417961ff')
ax1[1].tick_params(axis='y', colors='#417961ff')
ax1[1].grid(False)
sns.pointplot(x="Mcut", y="Log-likelihood Score", data=Dataframe_mag,hue="model",palette=['#cb6952ff','#36423cff'],ax=ax4)
ax4.yaxis.tick_left()
ax4.yaxis.set_label_position("left")
ax4.set_ylabel('').set_visible(False)
ax4.legend(fontsize=18)
ax1[1].legend().set_visible(False)

fig.subplots_adjust(top=0.953,
bottom=0.165,
left=0.063,
right=0.901,
hspace=0.205,
wspace=0.141)



######## Parameter Plot - uncomment to plot ETAS parameters

# figparams, axparams = plt.subplots(5,1,sharex=True)
# axparams[0].plot(train_Mcuts,ETAS_mu,linewidth=3)
# axparams[0].set_ylabel(r'$\mu$')

# if earthquake_for_partition == 'Visso':
# 	ETAS_k0[16]=np.nan
# 	ETAS_k0[14]=np.nan
# 	ETAS_k0[13]=np.nan
# 	ETAS_k0[12]=np.nan

# axparams[1].plot(train_Mcuts,ETAS_k0,linewidth=3)
# axparams[1].set_ylabel(r'$k_0$')

# axparams[2].plot(train_Mcuts,ETAS_a,linewidth=3)
# axparams[2].set_ylabel(r'$a$')

# axparams[3].plot(train_Mcuts,ETAS_c,linewidth=3)
# axparams[3].set_ylabel(r'$c$')

# axparams[4].plot(train_Mcuts,ETAS_omega,linewidth=3)
# axparams[4].set_ylabel(r'$p$')
# axparams[4].set_xlabel('Mcut')
# axparams[4].set_xticks(np.arange(lowestMcut, 3.1, 0.1))



plt.show()

