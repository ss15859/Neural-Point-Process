import numpy as np
import scipy.stats
import pandas as pd

from scipy.optimize import minimize


def collate_likelihoods(Lvec,Boolvec):

    collated_vec = np.zeros(sum(Boolvec))
    count = 0
    k=0

    for i in range(len(Lvec)):
        if Boolvec[i]:
            collated_vec[count] = sum(Lvec[i-k:i+1])
            k=0
            count+=1
            
        else:
            k+=1
            
    return collated_vec

def collate_times(dT_dat,dM_dat,Mtarget):

        k=0
        dT_dat_targets = np.ones_like(dT_dat)
        mask = np.ones_like(dT_dat,dtype=bool)

        for i in range(len(dT_dat)):
            if(dM_dat[i]>=Mtarget):
                dT_dat_targets[i] = sum(dT_dat[i-k:i+1])
                k=0

            else:
                dT_dat_targets[i] = sum(dT_dat[i-k:i+1])
                k+=1
                mask[i]=False
                
        return dT_dat_targets, mask 


def datetime_to_hours(Catalog):

    Catalog['datetime'] = pd.to_datetime(Catalog[['year', 'month', 'day', 'hour', 'minute','second']])
    Catalog['time'] = (Catalog['datetime']-Catalog['datetime'][0])/ pd.to_timedelta(1, unit='H')
    return Catalog

def truncate_catalog_by_threshold(Catalog,Mcut):

    sub_catalog = Catalog[Catalog.mw > Mcut]
    
    return  sub_catalog

def append_burn_in_to_test_set(T_train,T_test,M_train,M_test,time_step):

    T_test = np.append(T_train[-time_step+1:],T_test) 
    M_test = np.append(M_train[-time_step+1:],M_test)
    return T_test, M_test

def find_Poisson_MLE(T_train, M_train, threshold):
    poissMLE = 1/np.ediff1d(T_train[M_train>=threshold]).mean()
    return poissMLE

def find_Poisson_likelihood_scores(poissMLE,T_test,M_test,threshold,time_step):

    PoissVec = np.log(poissMLE)*(M_test[1:]>=threshold)- poissMLE*(T_test[1:]-T_test[:-1])
    PoissVec = PoissVec[time_step+1:]

    boolvec = M_test[time_step+1:]>=threshold
    PoissVec = collate_likelihoods(PoissVec,boolvec)

    return PoissVec

def fmd(mag,mbin):

    mag = np.array(mag)
    
    mi = np.arange(min(np.round(mag/mbin)*mbin), max(np.round(mag/mbin)*mbin),mbin)
    
    nbm = len(mi)
    cumnbmag = np.zeros(nbm)
    nbmag = np.zeros(nbm)

    for i in range(nbm):
        cumnbmag[i] = sum((mag > mi[i]-mbin/2))

    cumnbmagtmp = np.append(cumnbmag,0)
    nbmag = abs(np.ediff1d(cumnbmagtmp))

    res = {'m':mi, 'cum':cumnbmag, 'noncum':nbmag}

    return res


def maxc(mag,mbin):

    FMD = fmd(mag,mbin)

    Mc = FMD['m'][np.where(FMD['noncum']==max(FMD['noncum']))[0]][0]

    return Mc