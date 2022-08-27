# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 20:03:42 2022

@author: 山抹微云
"""

"""
TODO
Loss
The Accuracy, Sensitivity, Precision, NDCG, Area Under Curve
"""

import numpy as np
import pandas as pd
import seaborn as sb

from sklearn.metrics import ndcg_score
from sklearn.preprocessing import OneHotEncoder
# In[]
df = pd.read_csv("database/data file/df_train_2003.csv")

x_data = df.loc[:, "1":"28"]
y_data = df.loc[:, "0"]

df_test = pd.read_csv("database/data file/df_test_2003.csv")

x_test_data = df_test.loc[:, "1":"28"]
y_test_data = df_test.loc[:, "0"]


from imblearn.over_sampling import SMOTE 
oversample = SMOTE() 
X_oversample, y_oversample = oversample.fit_resample(x_data, y_data) 
X_test_oversample, y_test_oversample = oversample.fit_resample(x_test_data, y_test_data) 
# In[]
import time
from DataReader import dataFeeder

from ESOA import ESOA

n_dim = 29
size_pop = 10
lb = -1*np.ones(n_dim)  
ub =  1*np.ones(n_dim)
max_iter = 1

batchsize = 10000
DF = dataFeeder(X_oversample, y_oversample, x_test_data, y_test_data, batchsize, threshold = 0.5)
s = time.time()
esoa = ESOA(DF, n_dim, size_pop, max_iter, lb, ub, rand = 160)
esoa.run()
e = time.time()
print("COST: ", e-s)

# In[]
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, precision_recall_curve

cal_x = x_test_data.copy()
cal_y = y_test_data.copy()

esoa_pred = DF.predict(esoa.best_para, cal_x.values)

import matplotlib.pyplot as plt

#esoa_pred[esoa_pred<0] = 0
#esoa_pred[esoa_pred>1] = 1
#esoa_pred = (esoa_pred+1)/2
esoa_fpr, esoa_tpr, esoa_thresold = roc_curve(cal_y.values, esoa_pred)



def graph_roc_curve_multiple(fpr, tpr):
    plt.figure(figsize=(10,8))
    plt.title('ROC Curve And AUC Of ESOA', fontsize=28)
    plt.plot(fpr, tpr, color='#F59B00')

    plt.axis([-0.01, 1, -0.01, 1.01])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=24)
    plt.ylabel('True Positive Rate', fontsize=24)
    plt.fill_between(fpr, tpr, step='post', alpha=0.2,
                     color='#F59B00')
    plt.legend()
    
graph_roc_curve_multiple(esoa_fpr, esoa_tpr)
plt.show()


precision, recall, pr_threshold = precision_recall_curve(cal_y.values, esoa_pred)

plt.step(recall, precision, color='r', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#F59B00')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])


from DataReader import *

def mapping(x, p=0.5):
    xc = x.copy()
    xc[xc>p] = 1
    xc[xc<=p] = 0
    return xc


prec = keras.metrics.Precision(name='precision')

tn, fn, fp, tp = confusion_matrix(cal_y, mapping(esoa_pred, p=0.5)).ravel()
sen = tp/(tp+fn)
acc_esoa = (tp+tn)/(tp+tn+fp+fn)
DF.auc.reset_states()
auc_esoa = roc_auc_score(cal_y.values, esoa_pred)
prec_esoa = tp/(tp+fp)

def ndcg(rel_true, rel_pred, p=None, form="linear"):
    """ Returns normalized Discounted Cumulative Gain
    Args:
        rel_true (1-D Array): relevance lists for particular user, (n_songs,)
        rel_pred (1-D Array): predicted relevance lists, (n_pred,)
        p (int): particular rank position
        form (string): two types of nDCG formula, 'linear' or 'exponential'
    Returns:
        ndcg (float): normalized discounted cumulative gain score [0, 1]
    """
    rel_true = np.sort(rel_true)[::-1]
    p = len(np.where(rel_true==1)[0])
    discount = 1 / (np.log2(np.arange(p) + 2))

    if form == "linear":
        idcg = np.sum(rel_true[:p] * discount)
        dcg = np.sum(rel_pred[:p] * discount)
    elif form == "exponential" or form == "exp":
        idcg = np.sum([2**x - 1 for x in rel_true[:p]] * discount)
        dcg = np.sum([2**x - 1 for x in rel_pred[:p]] * discount)
    else:
        raise ValueError("Only supported for two formula, 'linear' or 'exp'")

    return dcg / idcg

ndcg_esoa = ndcg(cal_y.values, mapping(esoa_pred, p=0.5))


print('#############################')
print('AUC(sk): ', auc_esoa*100)
print('ACC: ', acc_esoa*100)
print('SEN: ', sen*100)
print('Prec: ', prec_esoa*100)
print('NDCG:', ndcg_esoa*100)
print('#############################')

# In[]
form = 'linear'
p = len(np.where(rel_true==1)[0])
rel_true = cal_y.values.copy()
rel_pred = mapping(esoa_pred, p=0.5).copy()

rel_true = np.sort(rel_true)[::-1]
p = min(len(rel_true), len(rel_pred), p)
discount = 1 / (np.log2(np.arange(p) + 2))

if form == "linear":
    idcg = np.sum(rel_true[:p] * discount)
    dcg = np.sum(rel_pred[:p] * discount)
elif form == "exponential" or form == "exp":
    idcg = np.sum([2**x - 1 for x in rel_true[:p]] * discount)
    dcg = np.sum([2**x - 1 for x in rel_pred[:p]] * discount)
else:
    raise ValueError("Only supported for two formula, 'linear' or 'exp'")

print(dcg/idcg)



# In[]

import requests
import matplotlib.pyplot as plt
import io
from PIL import Image
# In[]

with open('index', 'rb') as f:
    temp = f.read()

temp_ = io.BytesIO(temp)
t = Image.open(temp_)
plt.imshow(t)




