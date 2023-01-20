from scipy.special import gammaincc, gammainccinv, gamma, gammainc
import numpy as np
import collections
import math
from scipy.optimize import minimize
import pandas as pd
from scipy import integrate

    
def estimate_beta_value(mags,threshold):

    beta = 1/(mags-threshold).mean()

    return beta

# expected number of aftershocks

def k(m,params,Mcut):
    
    if(isinstance(m, (list, tuple, np.ndarray,pd.Series))):
        x = np.where(m>=Mcut,params['k0']*np.exp(params['a']*(m-params['M0'])),0)
        
    else:
        if(m>=Mcut):
            x = params['k0']*np.exp(params['a']*(m-params['M0']))

        else:
            x= 0
               
    return x


# # omori decay kernel


def f(x,params):
    return (params['omega'] - 1) * params['c']**(params['omega'] - 1) * 1/((x + params['c'])**params['omega'])

#integrated omori kernel

def H(t,params):


    if(isinstance(t, (list, tuple, np.ndarray,pd.Series))):
            x = np.where(t>=0,1 - params['c']**(params['omega'] - 1)/(t + params['c'])**(params['omega'] - 1),0)
            
    else:
        if(t>=0):
            x = 1 - params['c']**(params['omega'] - 1)/(t + params['c'])**(params['omega'] - 1)

        else:
            x= 0



    return x



def likelihood(Tdat,Mdat,maxtime,params,time_step):

    M0 = params['M0']

    temp=0
    
    for i in range(time_step+1,len(Tdat)):
        
        temp += np.log((params['mu'] + sum(k(Mdat[:(i )],params,M0) * f(Tdat[i] - Tdat[:(i)],params)))+1e-15)

    vec = Tdat[time_step+1:]
    vecM = Mdat[time_step+1:]

    index = time_step

    temp = temp - params['mu']*(maxtime-Tdat[index].min())
    temp = temp - (sum(k(Mdat,params,M0) * H(maxtime - Tdat,params))-sum(k(Mdat[:index],params,M0) * H(maxtime - Tdat[:index],params)))


    return temp/len(vec)


def likelihood_scores(Tdat,Mdat,maxtime,Mcut,M0pred,params,time_step):
    vec = np.zeros_like(Tdat)
    
    M0 = params['M0']
    fd = np.exp(-params['beta']*(M0pred-Mcut))
    

    for i in range(1,len(vec)):
        vec[i]+= np.log(fd*intensity(Tdat[i],[Tdat[:i],Mdat[:i]],params,M0))
        j=0
        if len(np.where(Mdat[:i]>=M0pred)[0])>=1:
            j = np.where(Mdat[:i]>=M0pred)[0][-1]
        vec[i]-= fd*params['mu']*(Tdat[i]-Tdat[j])
        if j==0:
            vec[i]-= fd*(sum(k(Mdat[:i],params,M0) * H(Tdat[i] - Tdat[:i],params)))
        else:
            vec[i]-= fd*(sum(k(Mdat[:i],params,M0) * H(Tdat[i] - Tdat[:i],params))-sum(k(Mdat[:j],params,M0) * H(Tdat[j] - Tdat[:j],params)))
        
    vec = vec[time_step+1:]
    vec = vec[Mdat[time_step+1:]>=M0pred]
        
    return vec

def magnitude_scores(T_test,M_test,Mcut,M0pred,params,time_step):

    ETASLLmarkvec = np.log(params['beta']*np.exp(-params['beta']*(M_test[M_test>=M0pred]-Mcut))*np.exp((M0pred-Mcut)*params['beta']))
    count = sum(M_test[:time_step+1]>=M0pred)
    ETASLLmarkvec = ETASLLmarkvec[count:]

    return ETASLLmarkvec

    
    
def intensity(t,hist,params,Mcut):
    
    if(t == hist[0][-1]):
        n = len(hist[0])-1

    else:
        n = len(hist[0])
    
    cumulative = 0
    
    for j in range(0,n):
        cumulative += k(hist[1][j],params,Mcut)*f(t-hist[0][j],params)


    lam = params['mu'] + cumulative
    return lam
    
    



def datetime64_to_days(dates):
    
    dates = np.array(dates)
    times = (dates-dates[0])/  np.timedelta64(1,'D')
    times=times.astype('float64')
    return times




def maxlikelihoodETAS(Tdat,Mdat,M0,maxtime=np.nan,initval=np.nan):
    
    if(math.isnan(maxtime)):
        maxtime = Tdat.max()

        beta = 1
        
    def fn(param_array):
        
        params = {
            'mu' : param_array[0],
            'k0' : param_array[1],
            'a' : param_array[2],
            'c' : param_array[3],
            'omega' : param_array[4],
            'M0' : M0,
            'beta': beta
        }
        
        if (params['mu'] <= 0 or params['omega'] <= 1 or params['a'] < 0 or params['k0'] < 0): 
            
            return(math.inf)
        
        val = -likelihood(Tdat,Mdat,maxtime=maxtime,time_step=0,params=params)
        
        return val
    
    
    if math.isnan(initval):
        initval = np.array([len(Tdat)/maxtime,0.5,0.5,1,2])
        
    else:
        initval = initval[:5]
    
    temp = minimize(fn,initval,method='Nelder-Mead',options={'xatol': 1e-4, 'disp': True,'maxiter':1e30})
    
    dic = { 'mu' : temp.x[0],
            'k0' : temp.x[1],
            'a' : temp.x[2],
            'c' : temp.x[3],
            'omega' : temp.x[4],
            'M0' : M0}
        
    
    return dic


