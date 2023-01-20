import csv
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

import seaborn as sns
sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2.2)

import math
import datetime
import ETAS
from ETAS import estimate_beta_value, maxlikelihoodETAS
from utils import truncate_catalog_by_threshold, datetime_to_hours, append_burn_in_to_test_set, find_Poisson_MLE, collate_likelihoods, find_Poisson_likelihood_scores
from NeuralPP import NPP

def main():

    type_of_data = str(sys.argv[1])
    Mcut = float(sys.argv[2])
    M0 = Mcut
    M0pred = 3.0
    time_step = 20

    synthetic_catalog = pd.read_csv('../data/Catalogs/synthetic_' + type_of_data + '_catalog.csv')
    synthetic_catalog['time'] = pd.to_datetime(synthetic_catalog['time'])
    synthetic_catalog = synthetic_catalog.sort_values('time')

    times = ETAS.datetime64_to_days(synthetic_catalog['time'])
    mags = np.array(synthetic_catalog['magnitude'])

    timeupto = select_training_testing_partition(type_of_data)

    T_train = times[times<timeupto]
    M_train = mags[times<timeupto]

    T_test = times[times>=timeupto]
    M_test = mags[times>=timeupto]


    T_train = T_train[M_train>=Mcut]
    M_train = M_train[M_train>=Mcut]

    T_test = T_test[M_test>=Mcut]
    M_test = M_test[M_test>=Mcut]

    T_test, M_test = append_burn_in_to_test_set(T_train,T_test,M_train,M_test,time_step)

    print('n train:' + str(T_train.shape))



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #  Training # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    ########### ETAS


    path_to_ETAS_params = '../data/ETAS_parameters/simulated_paramsMcut-'+str(Mcut)+'_' + type_of_data + '.csv'

    # if True:
    if not os.path.isfile(path_to_ETAS_params):

        print('Findling MLE parameters')

        start_time = datetime.datetime.now()

        MLE = maxlikelihoodETAS(T_train,M_train,M0=M0)

        end_time = datetime.datetime.now()

        MLE['train_time'] = start_time - end_time

        ETAS_train_time = MLE['train_time']

        MLE['beta'] = estimate_beta_value(M_test,M0)

        with open(path_to_ETAS_params, 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in MLE.items():
               writer.writerow([key, value])

    else:
        with open(path_to_ETAS_params) as csv_file:
            reader = csv.reader(csv_file)
            MLE = dict(reader)

            ETAS_train_time = MLE['train_time']
            del[MLE['train_time']]

            MLE = {k:float(v) for k, v in MLE.items()}

    print(MLE)  

    ############# Neural Network

    checkpoint = 'simulated_checkpoint_'+str(Mcut)+'_'+type_of_data +str(M0pred)+str(time_step)

    # if True:
    if not os.path.isfile('../data/Checkpoints/'+ checkpoint+'.index'):
        
        print('Training Network')

        start_time = datetime.datetime.now()

        npp = NPP(time_step=time_step,size_rnn=64,size_nn=64,size_layer_chfn=3,size_layer_cmfn=3,M0pred=M0pred).set_train_data(T_train,M_train).set_model(0).compile(lr=1e-3).fit_eval(epochs=400,batch_size=256,plot_training=False).save_weights(checkpoint)
        
        end_time = datetime.datetime.now()

        npp.train_time = start_time - end_time
    else:
        print('Retreiving Network')
        npp = NPP(time_step=time_step,size_rnn=64,size_nn=64,size_layer_chfn=3,size_layer_cmfn=3,M0pred=M0pred).set_train_data(T_train,M_train).set_model(0,stateful=False).load_weights(checkpoint)
        npp.train_time = 0

    ############ Poisson

    poissMLE = find_Poisson_MLE(T_train,M_train,M0pred)





    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #  Testing  # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    path_to_results = '../data/Results/'+ type_of_data + '_simulated_resultsMcut-'+str(Mcut)+'_M0pred:'+str(M0pred)+'.csv'

    # if True:
    if not os.path.isfile(path_to_results):

        print('calculating likelihoods')

    ##### Poisson   

        Poisson_time_scores = find_Poisson_likelihood_scores(poissMLE,T_test,M_test,M0pred,time_step)

        LLpoiss = Poisson_time_scores.mean()

        print('Poisson Likelihood:   '+str(LLpoiss))

    ##### ETAS

        time_of_last_target_event = T_test[M_test>=M0pred].max() 

        ETAS_time_scores = ETAS.likelihood_scores(T_test,M_test,maxtime=time_of_last_target_event,Mcut=Mcut,M0pred=M0pred,params=MLE,time_step=time_step)

        print('ETAS Temporal Likelihood:   ' + str(ETAS_time_scores.mean()))

        ETAS_mag_scores = ETAS.magnitude_scores(T_test,M_test,Mcut=Mcut,M0pred=M0pred,params=MLE,time_step=time_step)

        print('ETAS Mark Likelihood:  '+str(ETAS_mag_scores.mean()))

    ##### NN

        npp.set_test_data(T_test,M_test).predict_eval(batch_size=512)

        print('NN Temporal Likelihood:   ' + str(npp.collated_LL.mean()))
        print('NN Mark Likelihood:  '+str(npp.LLmag.mean()))

    else: # read in results

        D = pd.read_csv(path_to_results)

        ETAS_time_scores = D.ETAS_pointwise_like
        print('ETAS Temporal Likelihood:   ' + str(ETAS_time_scores.mean()))

        npp.collated_LL = D.NN_pointwise_lik
        print('NN Temporal Likelihood:   ' + str(npp.collated_LL.mean()))

        LLpoiss = D.LLpoiss[0]
        print('Poisson Likelihood:   '+str(LLpoiss))

        ETAS_mag_scores = D.ETASLLmarkvec
        print('ETAS Mark Likelihood:  '+str(ETAS_mag_scores.mean()))
        npp.LLmag = D.nppLLmag
        print('NN Mark Likelihood:  '+str(npp.LLmag.mean()))

        npp.train_time = D.NN_train_time[0]

    ####################################################################### Generate output file 

    npredictions = ETAS_time_scores.shape[0]


    d ={'ETAS_pointwise_like':ETAS_time_scores,
        'NN_pointwise_lik':npp.collated_LL,
        'ETASLLmarkvec':ETAS_mag_scores,
        'nppLLmag':npp.LLmag, 
        'LLpoiss': np.repeat(LLpoiss,npredictions),
        'ETAS_mu':np.repeat(MLE['mu'],npredictions),
        'ETAS_k0':np.repeat(MLE['k0'],npredictions),
        'ETAS_a':np.repeat(MLE['a'],npredictions),
        'ETAS_c':np.repeat(MLE['c'],npredictions),
        'ETAS_omega':np.repeat(MLE['omega'],npredictions),
        'ETAS_train_time': np.repeat(ETAS_train_time,npredictions),
        'NN_train_time': np.repeat(npp.train_time,npredictions),
        'ntrain': np.repeat(T_train.shape[0],npredictions),
        'time_step' : np.repeat(time_step,npredictions)
        }

    if not os.path.isfile(path_to_results):

        print('writing data')

        D = pd.DataFrame(d)

        D.to_csv(path_to_results)



    if len(sys.argv) > 3:
        if sys.argv[3] == 'plot_mag_densities':

            chopped_times=T_test[T_test>timeupto]
            chopped_mags =M_test[T_test>timeupto]
            times_of_targets = chopped_times[M_test[T_test>timeupto]>=M0pred]
            mags_of_targets = chopped_mags[M_test[T_test>timeupto]>=M0pred]

            x = np.linspace(2,7,200)
            my_dpi=96
            fig, ax = plt.subplots(1,2,figsize=(1224/my_dpi, 639/my_dpi), dpi=my_dpi)

            target_index = int(np.where(mags_of_targets==mags_of_targets.max())[0])
            index = np.where(M_test==M_test.max())
            index = int(index[0])

            for i in range(2):


                T = T_test[:index]
                M = M_test[:index]
                new_time = T_test[index]

                reshape_RNN_times,reshape_RNN_mags, reshape_CHFN, reshape_CMFN, mask  = npp.magreshape([T,M],new_time)

                yNN = [npp.magdensfunc(i,reshape_RNN_times,reshape_RNN_mags, reshape_CHFN, reshape_CMFN,mask) for i in x]

                y = MLE['beta']*np.exp(-MLE['beta']*(x-Mcut))*np.exp((M0pred-Mcut)*MLE['beta'])*(x>M0pred)

                axi = np.ravel(ax)[i]

                axi.set_ylim([0,3.5])
                axi.fill(x,yNN,label = 'NN',color='#cb6952ff',linewidth=3,alpha=0.5,zorder=1000)
                axi.fill(x,y,label = 'ETAS',color = '#36423cff',linewidth=3,alpha=0.7)
                
                axi.scatter(mags_of_targets[target_index],0,clip_on=False,zorder=100000,color = '#be2a35',s=120,label='Observed Magnitude')
                axi.set_xlabel('Mw',fontsize=17)
                axi.set_ylabel('density',fontsize=17)
                axi.tick_params(axis='both', which='major', labelsize=12)

                index += 1
                target_index+=1


            ax[0].legend(fontsize=15)

            fig.subplots_adjust(top=0.927,
            bottom=0.157,
            left=0.089,
            right=0.968,
            hspace=0.2,
            wspace=0.225)

            plt.show()


def select_training_testing_partition(type_of_data):
    if type_of_data == 'complete':
        timeupto = 4000
    elif type_of_data == 'incomplete':
        timeupto = 4000
    else:
        print('Invalid argument')
        sys.exit()

    return timeupto




if __name__ == '__main__':
    main()
