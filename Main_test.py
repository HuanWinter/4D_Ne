########################## import modules ###############################

import numpy as np
import os
import sys

import netCDF4 as nc
import linecache as lc
import pandas as pd
import time
from scipy import io as sio
import progressbar
import torch
from shutil import copyfile
import json
import torch
import ipdb

# import Src
sys.path.append("Src")
from Py_Fun import Project, Print_test, Te_IRI_obj, network_develop

if not os.path.isdir('Projects/'):
    os.mkdir('Projects')

########################## Read Configurations ###############################
#%matplotlib notebook

# Read Config file
with open('Config.json') as json_data_file:
    Config = json.load(json_data_file)

X_select_keys = [key for key in Config['X_select'].keys()]
X_select_values = [value for value in Config['X_select'].values()]

Y_select_keys = [key for key in Config['Y_select'].keys()]
Y_select_values = [value for value in Config['Y_select'].values()]

Stations = [key for key in Config['Test_num'].keys()]
Test_nums = [value for value in Config['Test_num'].values()]

Format_keys = [key for key in Config['Figure_format'].keys()]
Format_values = [value for value in Config['Figure_format'].values()]

# Read Global Coefficients
#with open('Global.json') as Global_coef:
#    Global = json.load(Global_coef)

# Read Project name, ISR datasets and thresholds for preprocessing
Project_name = Config['Names']['Project']
ISR_name = Config['Names']['ISR']
ISR_range = Config['Para_range']

# Read which variable has been picked
index_X = []
index_Y = []


#ipdb.set_trace()

for i in range(len(X_select_keys)):
    if X_select_values[i]:
        #ipdb.set_trace()
        index_X.append(i)
for i in range(len(Y_select_keys)):
    if Y_select_values[i]:
        index_Y.append(i)

########################## Project ###############################

project = Project(Project_name)
project.create()

# remove the project
#project.remove()

Print_test(ISR_range)
time.sleep(1)

num=50
num_init =0

'''
X = []
Y = []
for i in range(num_init,num_init+num):
    print(['data/XY_'+str(i)+'.mat'])
    temp = []
    #print(sio.loadmat(['data/XY_'+str(i)+'.mat']))
    temp = sio.loadmat('data/XY_'+str(i)+'.mat')
    X_temp = temp['X']
    Y_temp = temp['Y']
    if i==num_init:
        X = X_temp
        Y = Y_temp
    else:
        X = np.vstack([X, X_temp])
        Y = np.vstack([Y, Y_temp])
    print(X.shape)
    print(Y.shape)

#print(np.argwhere(np.isnan(X)))
#print(~np.isnan(X[7]))
'''
if Config['Flags']['Preprocess']:

    # Read and precprocess ISR data
    Out = project.Preprocess(X.T,Y.T)
    X = Out[0]
    Y = Out[1]

    index_X = np.asarray(index_X).squeeze()
    index_Y = np.asarray(index_Y).squeeze()

    Print_test(index_Y)
    Print_test(Y.shape)

    # Variable Selection
    X_t = X[index_X,:]
    Y_t = Y[index_Y,:]

    #index = (~np.isnan(X[:,7]) & (Y[:,1]>=1e4) & (X[:,0]>200))
    #X_t = X[index,:]
    #Y_t = Y[index,1]

    #print(np.argwhere(np.isnan(X_temp)).shape)

    #import ipdb; ipdb.set_trace()

    # Normlisation
    X,Y = project.Normlise(X_t, Y_t)

    Y = Y.unsqueeze(0)

    _ , _ = project.Dataset_create(X.T, Y.T)
    #Print_test(Out_train)
    #Print_test(Out_test)
    time.sleep(.5)

########################## Modelling ##################################
# to make it simple, change index_Y.shape to 1
if Config['Flags']['Modelling']:
    #Print_test(len(index_Y))
    net = project.Modelling(len(index_X), 1)
    Print_test(net)
    time.sleep(.5)

    print(X.shape)
    print(Y.shape)

########################## Statistical Accuracy ##################################
if Config['Flags']['Accuracy']:

    Acc = np.zeros(len(Stations))
    for i in range(len(Stations)):
        Acc[i] = project.Accuracy(i)
    Print_test(Acc)
    #sio.savemat()
    time.sleep(.5)
