# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:39:53 2022

@author: 山抹微云
"""


import numpy as np
import pandas as pd


import tensorflow.keras as keras
from sklearn.metrics import roc_auc_score, precision_recall_curve



# In[]
from sklearn.metrics import log_loss, confusion_matrix

class dataFeeder:
    def __init__(self, X, Y, x_validation, y_validation, batchsize, cost_matrix = [0, 5, 1, 0], threshold = 0.5):

        self.batchsize = batchsize
        if type(X) is not pd.core.frame.DataFrame:
            self.X_train, self.y_train = X, Y
        else:
            self.X_train, self.y_train = X.values, Y.values
            
        if type(x_validation) is not pd.core.frame.DataFrame:
            self.X_test, self.y_test = x_validation, y_validation
        else:
            self.X_test, self.y_test = x_validation.values, y_validation.values

        self.gen_batch()
        
        self.cost_matrix = cost_matrix
        self.acc  = keras.metrics.BinaryAccuracy(name='accuracy')

        self.tp   = keras.metrics.TruePositives(name='tp',thresholds=threshold)
        self.fp   = keras.metrics.FalsePositives(name='fp',thresholds=threshold)
        self.tn   = keras.metrics.TrueNegatives(name='tn',thresholds=threshold)
        self.fn   = keras.metrics.FalseNegatives(name='fn',thresholds=threshold)

        self.prec = keras.metrics.Precision(name='precision',thresholds=threshold)
        self.rec  = keras.metrics.Recall(name='recall')
        self.auc  = keras.metrics.AUC(name='auc')
        self.prc  = keras.metrics.AUC(name='prc', curve='PR')
        
        
    def refill(self, V, batch):
            V = np.tile(V, (batch, 1))
            return V
    
    def cal_cost(self, tn, fp, fn, tp, y_true, y_pred):

        cost_loss = (tn*self.cost_matrix[0]+fp*self.cost_matrix[1] + fn*self.cost_matrix[2] + tp*self.cost_matrix[3])    	
        return cost_loss/self.batchsize
    
    def gen_batch(self):

        batch_ind = np.random.choice(np.arange(len(self.X_train)), self.batchsize, replace=False)
        self.x_batch = self.X_train[batch_ind, :]
        self.y_batch = self.y_train[batch_ind]

        #batch_ind = np.random.choice(np.arange(len(self.X_test)), int(len(self.X_test)/4), replace=False)
        self.x_test_batch = self.X_test#[batch_ind, :]
        self.y_test_batch = self.y_test#[batch_ind]

    
    def loss(self, w):
        y_ = np.sum(self.refill(w[:-1], self.batchsize)*self.x_batch, axis=1)+w[-1]
        y_[y_<-100] = -100

        y = np.tanh(y_)
        y = (y+1)/2

        #L1 =  log_loss(self.y_batch.reshape(-1, 1), np.column_stack([y, 1-y]))
        
        self.resetState()
        self.tp.update_state(self.y_batch, y)
        self.fp.update_state(self.y_batch, y)
        self.tn.update_state(self.y_batch, y)
        self.fn.update_state(self.y_batch, y)
        self.acc.update_state(self.y_batch, y)
        self.auc.update_state(self.y_batch, y)
        self.prec.update_state(self.y_batch, y)
        tp_esoa   = self.tp.result().numpy()
        fp_esoa   = self.fp.result().numpy()
        tn_esoa   = self.tn.result().numpy()
        fn_esoa   = self.fn.result().numpy()
        acc_esoa  = self.acc.result().numpy()
        auc_esoa  = roc_auc_score(self.y_batch, y)#self.auc.result().numpy()
        prec_esoa = self.prec.result().numpy()
        
        sen_esoa = tp_esoa / (tp_esoa + fn_esoa)
        
        L1 =  self.cal_cost(tn_esoa, fp_esoa, fn_esoa, tp_esoa, self.y_batch, y)
        L2 = 1/(auc_esoa) + 1/(prec_esoa+np.spacing(1))# + 1/(sen_esoa+np.spacing(1))
        L3 = np.sum(w**2)

        L = 1 * L1 +  L2 + 0.5 * L3

        return L
    
    def validation(self, w):
        
        y_ = np.sum(self.refill(w[:-1], len(self.x_test_batch))*self.x_test_batch, axis=1)+w[-1]
        y_[y_<-100] = -100
        
        y = np.tanh(y_)
        y = (y+1)/2
        self.resetState()
        self.tp.update_state(self.y_test_batch, y)
        self.fp.update_state(self.y_test_batch, y)
        self.tn.update_state(self.y_test_batch, y)
        self.fn.update_state(self.y_test_batch, y)
        self.acc.update_state(self.y_test_batch, y)
        self.auc.update_state(self.y_test_batch, y)
        self.prec.update_state(self.y_test_batch, y)
        tp_esoa   = self.tp.result().numpy()
        fp_esoa   = self.fp.result().numpy()
        tn_esoa   = self.tn.result().numpy()
        fn_esoa   = self.fn.result().numpy()
        acc_esoa  = self.acc.result().numpy()
        auc_esoa  = roc_auc_score(self.y_test_batch, y)#self.auc.result().numpy()
        prec_esoa = self.prec.result().numpy()
        
        sen_esoa = tp_esoa / (tp_esoa + fn_esoa)
        print('#############################')
        print('---------Validation----------')
        print('AUC: ', auc_esoa)
        print('ACC: ', acc_esoa)
        print('SEN: ', sen_esoa)
        print('Prec: ', prec_esoa)
        print('#############################')

        oe = auc_esoa + 0* prec_esoa - (1 - acc_esoa) - np.abs(0.2 - prec_esoa)
        
        L1 =  self.cal_cost(tn_esoa, fp_esoa, fn_esoa, tp_esoa, self.y_batch, y)
        L2 = 1/(auc_esoa) + 1/(prec_esoa+np.spacing(1))
        L3 = np.sum(w**2)

        L = 1 * L1 +  L2 + 0.5 * L3
        
        return oe
    
    def predict(self, w, x_test_batch):
        
        y_ = np.sum(self.refill(w[:-1], len(x_test_batch))*x_test_batch, axis=1)+w[-1]
        y_[y_<-100] = -100

        y = np.tanh(y_)
        y = (y+1)/2
        return y

    def proof(self, w):

        y_ = np.sum(self.refill(w[:-1], len(self.X_test))*self.X_test, axis=1)+w[-1]
        y_[y_<-100] = -100

        y = np.tanh(y_)
        y = (y+1)/2
        self.resetState()
        self.tp.update_state(self.y_test, y)
        self.fp.update_state(self.y_test, y)
        self.tn.update_state(self.y_test, y)
        self.fn.update_state(self.y_test, y)
        self.acc.update_state(self.y_test, y)
        self.auc.update_state(self.y_test, y)
        self.prec.update_state(self.y_test, y)
        tp_esoa = self.tp.result().numpy()
        fp_esoa = self.fp.result().numpy()
        tn_esoa = self.tn.result().numpy()
        fn_esoa = self.fn.result().numpy()
        acc_esoa = self.acc.result().numpy()
        auc_esoa = self.auc.result().numpy()
        prec_esoa = self.prec.result().numpy()
        
        sen_esoa = tp_esoa / (tp_esoa + fn_esoa)
        print('#############################')
        print('---------Validation----------')
        print(y)
        print('AUC: ', auc_esoa)
        print('ACC: ', acc_esoa)
        print('SEN: ', sen_esoa)
        print('Prec: ', prec_esoa)
        print('#############################')

        
        return y
    
    def resetState(self):
        self.tp.reset_states()
        self.fp.reset_states()
        self.tn.reset_states()
        self.fn.reset_states()
        self.acc.reset_states()
        self.auc.reset_states()
        self.prec.reset_states()

