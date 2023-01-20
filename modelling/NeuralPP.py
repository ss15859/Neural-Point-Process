# Code adapted from: 
# T. Omi, N. Ueda, and K. Aihara, 
# "Fully neural network based model for general temporal point processes", 
# Advances in Neural Information Processing Systems 32 (Neurips 2019), 2120 (2019).
# https://github.com/omitakahiro/NeuralNetworkPointProcess

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

tf.random.set_seed(1)


import matplotlib.pyplot as plt


class NPP():
    
    def __init__(self,time_step,size_rnn,size_nn,size_layer_chfn,size_layer_cmfn,M0pred):
        self.time_step = time_step
        self.size_rnn = size_rnn
        self.size_nn = size_nn
        self.size_layer_chfn = size_layer_chfn
        self.size_layer_cmfn = size_layer_cmfn
        self.M0pred= M0pred
        
      

    def collate_times(self,dT_dat,dM_dat,Mtarget):
    
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
        
    def set_train_data(self,times,mags):


        ## format the input data
        
        self.T_train=times
        self.M_train=mags
        
        # remove first magnitude since our input is time intervals
        
        dM_train = np.delete(mags,0)

        dT_train = np.ediff1d(times) # transform a series of timestamps to a series of interevent intervals: T_train -> dT_train
        n = dT_train.shape[0]
        n2 = dM_train.shape[0]

        print(dT_train.shape)

        dT_train_targets, train_mask = self.collate_times(dT_train,dM_train,self.M0pred)
        
        
        input_RNN_times = np.array( [ dT_train[i:i+self.time_step] for i in range(n-self.time_step) ]).reshape(n-self.time_step,self.time_step,1)
        input_RNN_mags = np.array( [ dM_train[i:i+self.time_step] for i in range(n2-self.time_step) ]).reshape(n2-self.time_step,self.time_step,1)
        
        self.input_RNN_times = input_RNN_times
        self.input_RNN_mags = input_RNN_mags
        self.input_CHFN = dT_train[-n+self.time_step:].reshape(n-self.time_step,1)
        self.input_CMFN =dM_train[-n+self.time_step:].reshape(n-self.time_step,1)
        self.train_mask = train_mask[-n+self.time_step:].reshape(n-self.time_step,1)
        
        
        return self
        
    def set_model(self,lam,batch_size=None,stateful=False):
        
        ## mean and std of the log of the inter-event interval and magnitudes, which will be used for the data standardization
        mu = np.log(np.ediff1d(self.T_train)).mean()
        sigma = np.log(np.ediff1d(self.T_train)).std()

        mu1 = np.log(self.M_train).mean()
        sigma1 = np.log(self.M_train).std()

        ## kernel initializer for positive weights
        def abs_glorot_uniform(shape, dtype=None, partition_info=None): 
            return K.abs(keras.initializers.glorot_uniform(seed=None)(shape,dtype=dtype))

        ## Inputs 
        time_history = layers.Input(shape=(self.time_step,1),batch_size=batch_size)
        mag_history = layers.Input(shape=(self.time_step,1),batch_size=batch_size)
        # event_history = layers.Input(shape=(self.time_step,2))
        elapsed_time = layers.Input(shape=(1,),batch_size=batch_size) # input to cumulative hazard function network (the elapsed time from the most recent event)
        current_mag = layers.Input(shape=(1,),batch_size=batch_size) # input to cumulative magnitude function
        mask = layers.Input(shape=(1,),batch_size=batch_size)


        ## log-transformation and standardization

        elapsed_time_nmlz = layers.Lambda(lambda x: (K.log(x)-mu)/sigma )(elapsed_time) 

        numpyA = np.array([[1/sigma,0],[0,1/sigma1]])

        numpyB = np.array([mu,mu1])
        numpyB = np.repeat(numpyB,self.time_step,axis=0)

        time_history_nmlz = layers.Lambda(lambda x: (K.log(x)-mu)/sigma )(time_history)
        mag_history_nmlz = layers.Lambda(lambda x: (K.log(x)-mu1)/sigma1 )(mag_history)

        event_history_nmlz = layers.Concatenate(axis=2)([time_history_nmlz,mag_history_nmlz])

        current_mag_nmlz = layers.Lambda(lambda x: (x-self.M0pred))(current_mag)

        ## RNN
        output_rnn = layers.LSTM(self.size_rnn,input_shape=(self.time_step,2),activation='tanh',stateful=stateful)(event_history_nmlz)

        ## the first hidden layer in the cummulative hazard function network
        hidden_tau = layers.Dense(self.size_nn,kernel_initializer=abs_glorot_uniform,kernel_constraint=keras.constraints.NonNeg(),use_bias=False,kernel_regularizer=regularizers.l2(lam))(elapsed_time_nmlz) # elapsed time -> the 1st hidden layer, positive weights
        hidden_rnn = layers.Dense(self.size_nn,kernel_regularizer=regularizers.l2(lam))(output_rnn) # rnn output -> the 1st hidden layer
        hidden = layers.Lambda(lambda inputs: K.tanh(inputs[0]+inputs[1]) )([hidden_tau,hidden_rnn])

        ## the second and higher hidden layers
        for i in range(self.size_layer_chfn-1):
            hidden = layers.Dense(self.size_nn,activation='tanh',kernel_initializer=abs_glorot_uniform,kernel_constraint=keras.constraints.NonNeg(),kernel_regularizer=regularizers.l2(lam))(hidden) # positive weights

        ## the first hidden layer in the cummulative magnitude function network
        hidden_mu = layers.Dense(self.size_nn,kernel_initializer=abs_glorot_uniform,kernel_constraint=keras.constraints.NonNeg(),use_bias=False,trainable=False,activation='relu')(current_mag_nmlz) # elapsed time -> the 1st hidden layer, positive weights
        hidden_rnn_mag = layers.Dense(self.size_nn)(output_rnn) # rnn output -> the 1st hidden layer
        hidden_mag = layers.Lambda(lambda inputs: K.tanh(inputs[0]+inputs[1]+inputs[2]) )([hidden_mu,hidden_rnn_mag,hidden_tau])

        ## the second and higher hidden layers
        for i in range(self.size_layer_cmfn-1):
            hidden_mag = layers.Dense(self.size_nn,activation='tanh',kernel_initializer=abs_glorot_uniform,kernel_constraint=keras.constraints.NonNeg())(hidden_mag) # positive weights



        ## Outputs
        Int_l = layers.Dense(1, activation='softplus',kernel_initializer=abs_glorot_uniform, kernel_constraint=keras.constraints.NonNeg() )(hidden) # cumulative hazard function, positive weights
        l = layers.Lambda( lambda inputs: K.gradients(inputs[0],inputs[1])[0] )([Int_l,elapsed_time]) # hazard function
        Int_l_mag = layers.Dense(1, activation='sigmoid',kernel_initializer=abs_glorot_uniform, kernel_constraint=keras.constraints.NonNeg() )(hidden_mag) # cumulative hazard function, positive weights
        l_mag= layers.Lambda( lambda inputs: K.gradients(inputs[0],inputs[1])[0] )([Int_l_mag,current_mag]) # hazard function

        ## define model
        self.model = Model(inputs=[time_history,mag_history,elapsed_time,current_mag,mask],outputs=[l,Int_l,l_mag,Int_l_mag])
        self.model.add_loss( -K.sum( (K.log( 1e-10 + l )*mask+ K.log(1e-10 + l_mag)*mask  - Int_l)) )# set loss function to be the negative log-likelihood function

        return self

    
    def compile(self,lr=1e-3):
        self.model.compile(keras.optimizers.Adam(lr=lr))
        return self
    

    def fit_eval(self,epochs=100,batch_size=256,plot_training=False,validation_split=0.2):
        
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,restore_best_weights=True)
        history = self.model.fit([self.input_RNN_times,self.input_RNN_mags,self.input_CHFN,self.input_CMFN,self.train_mask],epochs=epochs,batch_size=batch_size,validation_split=validation_split,callbacks=[es],shuffle=False)
        self.best_val_loss =  min(history.history['val_loss'])
        
        if plot_training:

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()
        
        return self


    def save_weights(self,path):
        self.model.save_weights('../data/Checkpoints/'+ path)
        return self

    def load_weights(self,path):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.model.load_weights('../data/Checkpoints/'+ path)
        return self 



    
    def set_test_data(self,times,mags):
        
        ## format the input data
        dM_test = np.delete(mags,0)
        dT_test = np.ediff1d(times)+1e-9
        n = dT_test.shape[0]
        n2 = dM_test.shape[0]

        dT_test_targets, test_mask = self.collate_times(dT_test,dM_test,self.M0pred)
        
        input_RNN_times = np.array( [ dT_test[i:i+self.time_step] for i in range(n-self.time_step) ]).reshape(n-self.time_step,self.time_step,1)
        input_RNN_mags = np.array( [ dM_test[i:i+self.time_step] for i in range(n2-self.time_step) ]).reshape(n2-self.time_step,self.time_step,1)
        
        self.input_RNN_times_test = input_RNN_times
        self.input_RNN_mags_test = input_RNN_mags
        self.input_CHFN_test = dT_test[-n+self.time_step:].reshape(n-self.time_step,1)
        self.input_CMFN_test =dM_test[-n+self.time_step:].reshape(n-self.time_step,1)
        self.test_mask = test_mask[-n+self.time_step:].reshape(n-self.time_step,1)
        
        return self
        
        
        

    def collate_likelihoods(self,Lvec,Boolvec):
    
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
            
        
    # predict and calculate the log-likelihood
    def predict_eval(self,batch_size=512):
        
        [self.lam,self.Int_lam,self.mag_dist,self.Int_mag_dist] = self.model.predict([self.input_RNN_times_test,self.input_RNN_mags_test,self.input_CHFN_test,self.input_CMFN_test,self.test_mask],batch_size=batch_size)
        self.LL = (np.log(self.lam+1e-10)*self.test_mask - self.Int_lam) 
        self.LLmag = (np.log(1e-10 + self.mag_dist ))[self.test_mask]
        self.LL_average = sum(self.LL)/sum(self.test_mask)
        self.collated_LL = self.collate_likelihoods(self.LL,self.test_mask)
        
        return self
    
    def eval_train_data(self,batch_size=512):
        [self.lam_train,self.Int_lam_train,self.mag_dist_train,self.Int_mag_dist_train] = self.model.predict([self.input_RNN_times,self.input_RNN_mags,self.input_CHFN,self.input_CMFN,self.train_mask],batch_size=batch_size)
        self.LL_train = (np.log(self.lam_train+1e-10)*self.train_mask - self.Int_lam_train) 
        self.LLmag_train = (np.log(1e-10 + self.mag_dist_train ))[self.train_mask]
        self.LL_average_train = sum(self.LL_train)/sum(self.train_mask)
        self.collated_LL_train = self.collate_likelihoods(self.LL_train,self.train_mask)
        
        return self
    
    def summary(self):
        return self.model.summary()



    def magreshape(self,hist,new_time):

        T, M = hist

        dM_test = np.delete(M,0)
        dT_test=np.append(T,new_time)
        dT_test = np.ediff1d(dT_test) 
        dM_test=np.append(dM_test,-1)
        n = dT_test.shape[0]
        n2 = dM_test.shape[0]
        input_RNN_times = np.array( [ dT_test[i:i+self.time_step] for i in range(n-self.time_step) ]).reshape(n-self.time_step,self.time_step,1)
        input_RNN_mags = np.array( [ dM_test[i:i+self.time_step] for i in range(n2-self.time_step) ]).reshape(n2-self.time_step,self.time_step,1)
        input_CHFN_test = dT_test[-n+self.time_step:].reshape(n-self.time_step,1)
        input_CMFN_test =dM_test[-n+self.time_step:].reshape(n-self.time_step,1)
        mask  = np.ones_like(input_CMFN_test)


        return input_RNN_times, input_RNN_mags, input_CHFN_test, input_CMFN_test, mask

    def magdensfunc(self,x,reshape_RNN_times,reshape_RNN_mags, reshape_CHFN, reshape_CMFN,mask):

            reshape_CMFN[-1] = x

            Int_m_test = self.model.predict([reshape_RNN_times,reshape_RNN_mags,reshape_CHFN,reshape_CMFN,mask],batch_size=reshape_RNN_times.shape[0])[2]

            return Int_m_test[-1]

    def magdistfunc(self,x,reshape_RNN, reshape_CHFN, reshape_CMFN,mask):

            reshape_CMFN[-1] = x

            Int_m_test = self.model.predict([reshape_RNN,reshape_CHFN,reshape_CMFN,mask],batch_size=reshape_RNN.shape[0])[3]

            return Int_m_test[-1]


    def timereshape(self,hist):

        T,M=hist
        dM_test = np.delete(M,0)
        dT_test = np.ediff1d(T)
        dT_test=np.append(dT_test,-1)
        dM_test=np.append(dM_test,0)
        n = dT_test.shape[0]
        n2 = dM_test.shape[0]

        dT_test_targets, test_mask = self.collate_times(dT_test,dM_test,self.M0pred)
        indexes_of_True = [i for i, x in enumerate(test_mask) if x]
        last_index = indexes_of_True[-1]

        
        input_RNN_times_test = np.array( [ dT_test[i:i+self.time_step] for i in range(n-self.time_step) ]).reshape(n-self.time_step,self.time_step,1)
        input_RNN_mags_test = np.array( [ dM_test[i:i+self.time_step] for i in range(n2-self.time_step) ]).reshape(n2-self.time_step,self.time_step,1)

        input_CHFN_test = dT_test[-n+self.time_step:].reshape(n-self.time_step,1)
        input_CMFN_test =dM_test[-n+self.time_step:].reshape(n-self.time_step,1)
        test_mask = test_mask[-n+self.time_step:].reshape(n-self.time_step,1)

        return input_RNN_times_test[-last_index:,:,:], input_RNN_mags_test[-last_index:,:,:], input_CHFN_test[-last_index:], input_CMFN_test[-last_index:], test_mask[-last_index:]

    def distfunc(self,x,reshape_RNN, reshape_CHFN, reshape_CMFN):

        reshape_CHFN = x

        int_l_test = self.model.predict([reshape_RNN,reshape_CHFN,reshape_CMFN],batch_size=512)[1]

        return 1-np.exp(-int_l_test[-1])

    def densfunc(self,x,reshape_RNN_times,reshape_RNN_mags, reshape_CHFN, reshape_CMFN,reshape_mask):

        reshape_CHFN[-1] = x

        lam, Int_lam = self.model.predict([reshape_RNN_times,reshape_RNN_mags,reshape_CHFN,reshape_CMFN,reshape_mask],batch_size=reshape_mask.shape[0])[:2]

        LL = (np.log(lam+1e-10)*reshape_mask - Int_lam) 

        likelihood = np.exp(utils.collate_likelihoods(LL,reshape_mask))

        return likelihood
    
    
