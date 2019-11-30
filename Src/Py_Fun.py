from shutil import copyfile
import shutil
import scipy.io as sio
import json
import torch
from numpy import linalg as LA
from torch.autograd import Variable
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
import scipy.io as sio
import numpy as np
import ipdb
from torch.utils.data import random_split
import os,sys,os.path
import string
import time
import madrigalWeb.madrigalWeb
import pickle
import filecmp

from datetime import datetime
import iri2016 as iri
from argparse import ArgumentParser
from matplotlib.pyplot import show
import iri2016.plots as piri
from multiprocessing import cpu_count, Pool
import progressbar

import h5py
from matplotlib import cm
from TIE_read import ncread,ncdump, TIE_grid_h_time
from netCDF4 import Dataset

################################### Global coefficients #################

with open('Config.json') as Config_para:
    #for line in json_data_file:
    #    data.append(json.loads(line))
    Config = json.load(Config_para)
    #Print_test(Global)

Sta_full_name = [value for value in Config['Names']['Sta_full_name'].values()]

X_select_keys = [key for key in Config['X_select'].keys()]
X_select_values = [value for value in Config['X_select'].values()]

Y_select_keys = [key for key in Config['Y_select'].keys()]
Y_select_values = [value for value in Config['Y_select'].values()]

Stations = [key for key in Config['Test_num'].keys()]
Test_nums = [value for value in Config['Test_num'].values()]

Format_keys = [key for key in Config['Figure_format'].keys()]
Format_values = [value for value in Config['Figure_format'].values()]

################################### Global Functions & Class #################

def Project_build(Project_path):

    os.mkdir(Project_path)
    os.mkdir(Project_path+'Coef/')
    os.mkdir(Project_path+'Figs/')
    os.mkdir(Project_path+'Data/')
    os.mkdir(Project_path+'History/')
    os.mkdir(Project_path+'Data/'+'Input')
    os.mkdir(Project_path+'Data/'+'Temporal')
    os.mkdir(Project_path+'Data/'+'Output')
    for i in range(len(Format_keys)):
        os.mkdir(Project_path+'Figs/'+Format_keys[i])
        for j in range(len(Stations)):
            os.mkdir(Project_path+'Figs/'+Format_keys[i]+'/'+Stations[j])
            os.mkdir(Project_path+'Figs/'+Format_keys[i]+'/'+Stations[j]+'/Events')
            os.mkdir(Project_path+'Figs/'+Format_keys[i]+'/'+Stations[j]+'/Seasons')

'''
    if not Config['Flags']['ISR_preprocess'] and not os.path.exists(Project_path+'Coef/'+'mean_std.mat'):
        copyfile('Data/mean_std.mat', Project_path+'Coef/'+'mean_std.mat')

    if not Config['Flags']['Modelling']:
        if not os.path.exists(Project_path+'Coef/'+'net.pkl'):
            copyfile('Data/net.pkl', Project_path+'Coef/'+'net.pkl')
    if not os.path.exists(Project_path+'Data/Temporal/'+'Train_Test.pickle'):
        copyfile('Data/Train_Test.pickle', Project_path+'Data/Temporal/'+'Train_Test.pickle')
    if not Config['Flags']['Comparison']:
        for i in range(len(Stations)):
            if not os.path.exists(Project_path+'Data/Output/'+Stations[i]+'_Te_outputs.mat'):
                copyfile('Data/'+Stations[i]+'_Te_outputs.mat',\
                     Project_path+'Data/Output/'+Stations[i]+'_Te_outputs.mat')

'''

class Print_test:
    """docstring for ."""
    def __init__(self, x, flag=False):
        if Config['Names']['Mode']=='test' or flag==True:
            print(x)
        else:
            if Config['Names']['Mode']=='main':
                #test_flag = False
                pass
            else:
                print('The mode from Config.json is incorrect, go with main mode, Please double check it')

################################### NN development #################

class network_develop:
    """docstring for network_develop."""
    def __init__(self, net):
        self.net = net
        self.Linear_num = 1
        self.Tanh_num = 1
        self.Sigmoid_num = 1
        self.ReLU_num = 1
        self.net = net
        self.layers = {
            'Tanh':'torch.nn.Tanh()',
            'ReLU':'torch.nn.ReLU()',
            'Sigmoid':'torch.nn.Sigmoid()'
        }

    def Tanh(self):
        #ipdb.set_trace()
        self.net.add_module('Tanh'+str(self.Tanh_num), eval(self.layers['Tanh']))
        self.Tanh_num = self.Tanh_num + 1

    def Sigmoid(self):
        self.net.add_module('Sigmoid'+str(self.Sigmoid_num), eval(self.layers['Sigmoid']))
        self.Sigmoid_num = self.Sigmoid_num + 1

    def ReLU(self):
        self.net.add_module('ReLU'+str(self.ReLU_num), eval(self.layers['ReLU']))
        self.ReLU_num = self.ReLU_num + 1

    def Linear(self, inputs, outputs):
        Print_test(outputs)
        self.net.add_module('dense'+str(self.Linear_num), torch.nn.Linear(inputs, outputs))
        self.Linear_num = self.Linear_num + 1

    def Export(self):
        net_out = self.net
        return net_out

################################### main phase #################

class Project:
    """docstring for ."""
    def __init__(self, Name):
        self.Name = Name
        self.Folder = os.getcwd()
        self.Project_folder = self.Folder+'/Projects/'
        self.Project_path = self.Folder+'/Projects/'+self.Name+'/'
        self.Mean_std_path = self.Folder+'/Projects/'+self.Name+'/Coef/mean_std.mat'
        self.ISR_path = self.Folder+'/Projects/'+self.Name+'/Data/Input/'+Config['Names']['ISR']+'.mat'
        self.Input_path = self.Folder+'/Projects/'+self.Name+'/Data/Input/'
        self.Output_path = self.Folder+'/Projects/'+self.Name+'/Data/Output/'
        self.Temporal_path = self.Folder+'/Projects/'+self.Name+'/Data/Temporal/'
        self.Figs_path = self.Folder+'/Projects/'+self.Name+'/Figs/'
        self.History_path = self.Folder+'/Projects/'+self.Name+'/History/'
        self.Train_Test_path = self.Folder+'/Projects/'+self.Name+'/Data/Temporal/Train_Test.pickle'
        self.net_path = self.Folder+'/Projects/'+self.Name+'/Coef/net.pkl'


    # project development
    def create(self):
        # establish the project
        if not os.path.isdir(self.Project_path):
            Project_build(self.Project_path)
            copyfile('Config.json', self.Project_path+'Config.json')
        else:
            flag = input("The project '"+self.Name+"' does exist. Would you like to overwrite it? (Y/N) ")
            if flag == 'Y' or flag == 'y':
                shutil.rmtree(self.Project_path)
                Project_build(self.Project_path)
                #ipdb.set_trace()
                print("The project '"+self.Name+"' has been overwritten.")
                copyfile('Config.json', self.Project_path+'Config.json')

            else:
                if flag == 'N' or flag == 'n':
                    print("Keep the project '"+self.Name+"'")
                    if not filecmp.cmp('Config.json', self.Project_path+'Config.json'):
                        i = 1
                        while os.path.exists(self.History_path+'Config_'+str(i)+'.json'):
                            i = i+1

                        shutil.move(self.Project_path+'Config.json', self.History_path+'Config_'+str(i)+'.json')
                        copyfile('Config.json', self.Project_path+'Config.json')
                        print('Config.json has been updated')
                    else:
                        print('no change has been found in Config file')
                else:
                    print("The project '"+self.Name+"' cannot be built.")
                    sys.exit()

    # remove the whole project
    def remove(self):
        shutil.rmtree(self.Project_path)
        print("The project '"+self.Name+"' has been removed.")

    def move(self, name, old_path, new_path):
        shutil.move(old_path+name, self.Project_path+new_path+name)
        print("The file "+ name +" has been removed to." + self.Project_path+new_path)

    def Preprocess(self, X, Y):

        #import ipdb; ipdb.set_trace()

        Altitude = X[0,:]
        Latitude = X[1,:]
        Longitude = X[2,:]
        Azi = X[3,:]
        DST = X[4,:]
        AE = X[5,:]
        AP = X[6,:]
        F107 = X[7,:]
        Kp = X[8,:]/10
        Vf = X[9,:]
        DoY = X[10,:]
        UT = X[11,:]
        hmF2 = X[12,:]
        NmF2 = X[13,:]

        Ne = Y[1,:]

        LT = UT+Longitude/15

        index = np.where(LT>=12)
        LT[index] = LT[index]-12
        index = np.where(LT<0)
        LT[index] = LT[index]+12

        #import ipdb; ipdb.set_trace()

        index = np.where(
                 (NmF2>Config['Para_range']['NmF2'][0]) & (NmF2<Config['Para_range']['NmF2'][1]) &\
                 (hmF2>Config['Para_range']['hmF2'][0]) & (hmF2<Config['Para_range']['hmF2'][1]) &\
                 (DoY>Config['Para_range']['DoY'][0]) & (DoY<Config['Para_range']['DoY'][1]) &\
                 (LT>Config['Para_range']['LT'][0]) & (LT<Config['Para_range']['LT'][1]) &\
#                 (VSH>Config['Para_range']['VSH'][0]) & (VSH<Config['Para_range']['VSH'][1]) &\
                 (Kp>Config['Para_range']['Kp'][0]) & (Kp<Config['Para_range']['Kp'][1]) &\
                 (F107>Config['Para_range']['F107'][0]) & (F107<Config['Para_range']['F107'][1]) &\
                 (Ne>Config['Para_range']['Ne'][0]) & (Ne<Config['Para_range']['Ne'][1]) &\
                 (Latitude>Config['Para_range']['Latitude'][0]) & (Latitude<Config['Para_range']['Latitude'][1]) &\
                 (Longitude>Config['Para_range']['Longitude'][0]) & (Longitude<Config['Para_range']['Longitude'][1]))#and X[0,:]<=500

        X_out = X[:,index].squeeze()
        Y_out = Y[:,index].squeeze()
        #Ref = X_out[[2,3,10,11,7],:]
        return X_out, Y_out

    # Normlization and save mean and std of X and Y
    def Normlise(self, X, Y, savemode = True):
        meanX = np.mean(X, axis=1)
        meanY = np.mean(Y)

        stdX = np.std(X, axis=1)
        stdY = np.std(Y)

        Print_test(meanX)
        Print_test(stdX)
        Print_test(X.shape)
        #ipdb.set_trace()

        X = torch.from_numpy((X - meanX[:,None]) / stdX[:,None]) #.type( torch.FloatTensor )
        Y = torch.from_numpy(Y - meanY) / stdY #.type( torch.FloatTensor )
        X = X.type(torch.FloatTensor)
        Y = Y.type(torch.FloatTensor )

        if savemode:
            sio.savemat(self.Mean_std_path, \
                    {'meanX':meanX, \
                     'meanY':meanY, \
                     'stdX':stdX, \
                     'stdY':stdY})
        return X,Y

    # Convert data to dataset
    def Dataset_create(self, X, Y, savemode=True):
        torch.manual_seed(Config['NN_parameters']['Seed_num']) # make sure modelling starts from a same space
        Batch_Size = Config['NN_parameters']['BATCH_SIZE']

        #当使用batch size训练数据时，首先将tensor转化为Dataset格式
        #ipdb.set_trace()
        torch_dataset = Data.TensorDataset(X, Y)
        train_size = int(Config['NN_parameters']['Train_percent'] * np.max(X.shape))
        test_size = int(np.max(X.shape) - train_size)
        #ipdb.set_trace()
        train_dataset, test_dataset = random_split(torch_dataset, [train_size, test_size])

        #将dataset放入DataLoader中
        loader_train = Data.DataLoader(
            dataset = train_dataset,
            batch_size = Batch_Size,#设置batch size
            shuffle = True,#打乱数据
            num_workers = Config['NN_parameters']['Num_workers']#多线程读取数据
        )

        #将dataset放入DataLoader中
        loader_test = Data.DataLoader(
            dataset = test_dataset,
            batch_size = test_size,#设置batch size
            shuffle = True,#打乱数据
            num_workers = Config['NN_parameters']['Num_workers']#多线程读取数据
        )
        dataset = []
        if savemode:
            dataset.append(loader_train)
            dataset.append(loader_test)
            with open(self.Train_Test_path, 'wb') as output:
                pickle.dump(dataset, output)

        return loader_train, loader_test

    #载入模型和参数
    def Accuracy(self, index, loader_test=[]):
        if len(loader_test)==0:
            with open(self.Train_Test_path, 'rb') as output:
                dataset = pickle.load(output)

            loader_test = dataset[1]
        for _, (X_test, Y_test) in enumerate(loader_test):
            pass

        std_mean = sio.loadmat(self.Mean_std_path)
        stdX = std_mean['stdX']
        stdY = std_mean['stdY']
        meanX = std_mean['meanX']
        meanY = std_mean['meanY']
        #import ipdb; ipdb.set_trace()

        if torch.cuda.is_available() and Config['NN_parameters']['GPU']:
            net = torch.load(self.net_path)
            X_test = X_test.cuda()
            Y_test = Y_test
        else:
            net = torch.load(self.net_path, map_location='cpu')


        #import ipdb; ipdb.set_trace()
        #获得载入模型的预测输出
        #entire_out = net(X_test[:,0:X_test.shape[1]])
        X_test1 = []
        X_test1 = X_test.cpu().numpy()*stdX+meanX
        X_lat = []
        X_lat = X_test1[:,1]

        Arec = np.where(np.abs(X_lat)<30)
        Mill = np.where(np.abs((X_lat>30)) & (np.abs(X_lat<60)))
        Poker = np.where(np.abs(X_lat>60))
        All = np.where(np.abs(X_lat<90))
        #print(len(label_y))
        #ipdb.set_trace()
        ind = eval(Stations[index])
        if torch.cuda.is_available() and Config['NN_parameters']['GPU']:

            entire_out = net(X_test[ind,:])
            # 获得当前softmax层最大概率对应的索引值
            pred = entire_out.cpu()
        else:
            pred = net(X_test[ind,:])
        #将二维压缩为一维
        pred_y = pred.data.numpy().squeeze()
        label_y = Y_test.data.numpy().T
        Print_test(label_y.shape)
        Print_test(pred_y.shape)
        Print_test(np.asarray(ind).shape[1])
        #plt.plot(range(1,Y.shape[1]+1), label_y, 'r.', markersize=0.5)
        #import ipdb; ipdb.set_trace()
        #plt.plot(range(1,Y.shape[1]+1), pred_y, 'b.', markersize=0.5)
        #fig.savefig(folder+str(epoch+1)+'_'+str(step/200+1)+'.png')
        RMSD = np.sqrt(np.sum((pred_y - label_y[0,ind])**2/np.asarray(ind).shape[1]))*stdY

        pred_y = pred_y*stdY+meanY
        label_y[0,ind] = label_y[0,ind]*stdY+meanY

        Rel_acc = np.mean(np.abs(pred_y - label_y[0,ind])/label_y[0,ind])
        print("Rel_acc= %.2f"%(Rel_acc))
        print("RMSD= %.2f"%(RMSD))
        return RMSD

    def Modelling(self, input_num, output_num, loader_train=[], loader_test = []):
        #3.利用自定义的前向传播过程设计网络，设置各层神经元数量
        # net = Net(input=2, hidden=10, output=2)
        # print("神经网络结构：",net)
        Print_test(self.Name)

        Print_test(self.net_path)
        Print_test(type(self.net_path))
        Print_test(os.path.exists(str(self.net_path)))

        if os.path.exists(self.net_path):
            if torch.cuda.is_available() and Config['NN_parameters']['GPU']:
                net = torch.load(self.net_path)
            else:
                net = torch.load(self.net_path, map_location='cpu')
        else:
            if Config['NN_parameters']['Inherit_Mode']:
                copyfile('Data/net.pkl', self.net_path)
                if torch.cuda.is_available() and Config['NN_parameters']['GPU']:
                    net = torch.load(self.net_path)
                else:
                    net = torch.load(self.net_path, map_location='cpu')
            else:
                net_init = torch.nn.Sequential()
                net_temp = network_develop(net_init)
                sta = input_num
                for i in range(len(Config['NN_parameters']['Hidden_layer'])):
                    bot = Config['NN_parameters']['Hidden_layer'][i]
                    net_temp.Linear(sta, bot)
                    eval('net_temp.'+Config['NN_parameters']['Activation_finction'][i]+'()')
                    sta = bot
                net_temp.Linear(sta, output_num)
                net = net_temp.Export()
                if torch.cuda.is_available() and Config['NN_parameters']['GPU']:
                    net = net.cuda()

        #ipdb.set_trace()
        ind_optim = Config['NN_parameters']['Optim'][Config['NN_parameters']['Optim_ind']]
        Print_test(ind_optim['Learning_rate'])
        cmd = 'torch.optim.' \
                + ind_optim['mode']+ '( net.parameters(), lr='\
                +str(ind_optim['Learning_rate'])+', momentum='\
                +str(ind_optim['momentum'])+', weight_decay='\
                +str(ind_optim['weight_decay']) +')'
        Print_test(cmd)
        optimizer = eval(cmd)
        Print_test(optimizer)
        loss_func = eval('torch.nn.'+Config['NN_parameters']['Loss']+'()')

        Print_test(loss_func)
        #ipdb.set_trace()
        if (len(loader_train) == 0):
            with open(self.Train_Test_path, 'rb') as output:
                dataset = pickle.load(output)

            loader_train = dataset[0]
            loader_test = dataset[1]

        std_mean = sio.loadmat(self.Mean_std_path)
        stdY = std_mean['stdY']
        if Config['NN_parameters']['GPU']:
            if torch.cuda.is_available():
                print('GPU is functional')
            else:
                print('GPU is unavailable, switch to CPU')

        losses = []
        #ipdb.set_trace()


        for _, (X_test, Y_test) in enumerate(loader_test):
            pass
        if torch.cuda.is_available() and Config['NN_parameters']['GPU']:
            X_test = X_test.cuda()
            Y_test = Y_test

        for epoch in range(Config['NN_parameters']['epoch_num']):
            for step, (batch_x, batch_y) in enumerate(loader_train):
                if torch.cuda.is_available() and Config['NN_parameters']['GPU']:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()

                out = net(batch_x)#输入训练集，获得当前迭代输出值
                #print(batch_x)
                #print(out)
                loss = loss_func(out, batch_y)#获得当前迭代的损失

                optimizer.zero_grad()#清除上次迭代的更新梯度
                loss.backward()#反向传播
                optimizer.step()#更新权重

                #ipdb.set_trace()


                if step%200==0:
                    
                    #loss_variance = np.diff(losses)
                    #if np.sum(loss_variance[-200:-1])>0

                    #ipdb.set_trace()

                    if torch.cuda.is_available() and Config['NN_parameters']['GPU']:
                        entire_out = net(X_test)
                        pred = entire_out.cpu()
                    else:
                        net = net.cpu()
                        pred = net(X_test)

                    pred_y = pred.data.numpy().squeeze()
                    label_y = Y_test.data.numpy().T
                    accuracy = np.sqrt(np.sum((pred_y - label_y)**2/Y_test.shape[0]))
                    print("第 %d 个epoch，第 %d 次迭代，RMSD为 %.2f"%(epoch+1, step/200+1, accuracy*stdY))
                    #在指定位置添加文本

            torch.save(net, self.net_path)

        return net

    def Predict(self, X, index_X):
        """
        model Te
        """

        std_mean = sio.loadmat(self.Mean_std_path)
        stdX = std_mean['stdX'].T
        stdY = std_mean['stdY'].T
        meanX = std_mean['meanX'].T
        meanY = std_mean['meanY'].T

        if torch.cuda.is_available() and Config['NN_parameters']['GPU']:
            net = torch.load(self.net_path)
        else:
            net = torch.load(self.net_path, map_location='cpu')
        #获得载入模型的预测输出
        #entire_out = net(X_test[:,0:X_test.shape[1]])
        X = X[index_X,:].squeeze()

        #meanX = meanX[index_X,].squeeze()
        #stdX = stdX[index_X,].squeeze()
        Print_test(meanX.shape)
        Print_test(stdX.shape)
        Print_test(['X.shape = ', X.shape])
        X = torch.from_numpy((X - meanX)/ stdX) #.type( torch.FloatTensor )
        Print_test(['X.shape = ', X.shape])

        if torch.cuda.is_available() and Config['NN_parameters']['GPU']:
        #print(len(label_y))
        #ipdb.set_trace()
            entire_out = net(X.type(torch.FloatTensor).T.cuda())
            # 获得当前softmax层最大概率对应的索引值
            pred = entire_out.cpu()
        else:
            pred = net(X.type(torch.FloatTensor).T)

        #将二维压缩为一维
        Print_test(['pred.shape = ', pred.shape])
        pred_y = pred.data.numpy().squeeze()*stdY+meanY
        return pred_y


    def Comparison(self):
        Print_test(self.Mean_std_path)
        std_mean = sio.loadmat(self.Mean_std_path)
        stdX = std_mean['stdX']
        stdY = std_mean['stdY']
        meanX = std_mean['meanX']
        meanY = std_mean['meanY']
        num = 0
        num_clu = num

        test_num = np.zeros(len(Stations))
        for i in range(len(Stations)):
            test_num[i] = Config['Test_num'][Stations[i]]
        Print_test(test_num)
        for i in range(len(Stations)):

            Test_path = self.Input_path + Stations[i] + '_LT_ref.mat'
            try:
                copyfile('Data/'+Stations[i]+'_LT_ref.mat', Test_path)
            except Exception as e:
                print('no such test dataset named ' + Stations[i] + ' in the Data/')

            data = sio.loadmat(Test_path)
            X = data['X_temp']
            Y = data['Y_temp']
            X_ref = data['X_ref']
            if X.shape[0]==11:
                X = np.vstack([X, X_ref[4,:]])

            NmF2 = X[0,:]
            hmF2 = X[1,:]
            Month = X[2,:]
            LT = X[3,:]
            VSH = X[4,:]
            Kp = X[5,:]
            F107 = X[6,:]
            Altitude = X[7,:]
            Ne = X[8,:]
            r = X[9,:]
            Latitude = X[10,:]
            Longitude = X[11,:]

            Ti = Y[0,:]
            Te = Y[1,:]

            index = np.where(
                     (NmF2>Config['Para_range']['NmF2'][0]) & (NmF2<Config['Para_range']['NmF2'][1]) &\
                     (hmF2>Config['Para_range']['hmF2'][0]) & (hmF2<Config['Para_range']['hmF2'][1]) &\
                     (Month>Config['Para_range']['Month'][0]) & (Month<Config['Para_range']['Month'][1]) &\
                     (LT>Config['Para_range']['LT'][0]) & (LT<Config['Para_range']['LT'][1]) &\
                     (VSH>Config['Para_range']['VSH'][0]) & (VSH<Config['Para_range']['VSH'][1]) &\
                     (Kp>Config['Para_range']['Kp'][0]) & (Kp<Config['Para_range']['Kp'][1]) &\
                     (F107>Config['Para_range']['F107'][0]) & (F107<Config['Para_range']['F107'][1]) &\
                     (Ne>Config['Para_range']['Ne'][0]) & (Ne<Config['Para_range']['Ne'][1]) &\
                     (r>Config['Para_range']['r'][0]) & (r<Config['Para_range']['r'][1]) &\
                     (Te>Config['Para_range']['Te'][0]) & (Te<Config['Para_range']['Te'][1]) &\
                     (Te>=Ti))#and X[0,:]<=500

            X = X[:,index].squeeze()[:,0:int(test_num[i])]
            Y = Y[1,index].squeeze()[0:int(test_num[i])]
            X_ref = X_ref[:,index].squeeze()[:,0:int(test_num[i])]

            if i == 0:
                X_t = X
                Y_t = Y
                Ref = X_ref
                num = num + len(Y)
                num_clu = np.hstack([num_clu, num])
            else:
                X_t = np.hstack([X_t, X])
                Y_t = np.hstack([Y_t, Y])
                Ref = np.hstack([Ref, X_ref])
                num = num + len(Y)
                num_clu = np.hstack([num_clu, num])

                Print_test(num_clu)
                Print_test(np.asarray(X_t).squeeze().shape)
                Print_test(np.asarray(Y_t).squeeze().shape)
                Print_test(np.asarray(Ref).squeeze().shape)

        return X_t, Y_t, Ref, num_clu

    def Show_case(self):
        sta_clu = Stations
        data_num = Config['Test_num']
        folder = self.Figs_path
        acc_TBT = np.zeros(len(sta_clu))
        acc_Brace = np.zeros(len(sta_clu))
        acc_model = np.zeros(len(sta_clu))

        for i in range(len(sta_clu)):
            Te = sio.loadmat(self.Output_path + sta_clu[i]+'_Te_outputs.mat')

            Te_TBT_t = Te['Te_TBT'].squeeze()
            Te_ISR_t = Te['Te_ISR'].squeeze()
            Te_model_t = Te['Te_DNN'].squeeze()
            Te_Brace_t = Te['Te_Brace'].squeeze()
            Te_ref = Te['Te_ref'].squeeze()
            data_num = Te['Te_num'].squeeze()
            index = []
            index_t = []
            index_t = np.where((Te_TBT_t<5000) \
                        & (Te_Brace_t<5000) \
                        & (Te_model_t<5000) \
                        & (Te_TBT_t>100) \
                        & (Te_Brace_t>100) \
                        & (Te_model_t>100) )

            Print_test(index_t)
            index = np.asarray(index_t).squeeze()
            Te_TBT_t = Te_TBT_t[index]
            Te_ISR_t = Te_ISR_t[index]
            Te_Brace_t = Te_Brace_t[index]
            Te_model_t = Te_model_t[index]
            Te_ref = Te_ref[:,index]

            label = np.diff(Te_ref[2,:])
            ind = np.where(label<0)
            ind = np.asarray(ind).squeeze()
            Print_test(ind.shape)
            Print_test(Te_ref.shape)
            Print_test(Te_ISR_t.shape)

            print('Start to generate Te events at '+ sta_clu[i] +'...')
            time.sleep(1)

            p = progressbar.ProgressBar()

            for j in p(range(ind.shape[0]-1)):
                bot = ind[j]+1
                top = ind[j+1]
                if top-bot<=5:
                    continue

                hour = int(Te_ref[6,top])
                minute = int((Te_ref[6,top] - hour)*60)
                second = int(((Te_ref[6,top] - hour)*60 - minute)*60)
                fig = plt.figure(figsize=(Config['Figure_size']))
                #ipdb.set_trace()
                plt.plot(Te_ISR_t[bot:top], Te_ref[2,bot:top], 'o-', markersize=Config['Font']['size']/2)
                plt.plot(Te_model_t[bot:top], Te_ref[2,bot:top], 'x-', markersize=Config['Font']['size']/2)
                plt.plot(Te_TBT_t[bot:top], Te_ref[2,bot:top], '*-', markersize=Config['Font']['size']/2)
                plt.plot(Te_Brace_t[bot:top], Te_ref[2,bot:top], '.-', markersize=Config['Font']['size'])
                plt.legend(['ISR', 'DNN', 'TBT-2012', 'Brace-78'],fontsize=Config['Font']['size'])
                plt.xlabel('Te(K)',fontdict=Config['Font'])
                plt.xticks(fontsize=Config['Font']['size'],fontweight=Config['Font']['weight'])
                plt.yticks(fontsize=Config['Font']['size'],fontweight=Config['Font']['weight'])

                if i == 0:
                    plt.ylabel(str(hour)+'LT\nAltitude(km)',fontdict=Config['Font'])
                else:
                    plt.ylabel('Altitude(km)',fontdict=Config['Font'])

                if hour==0:
                    plt.title(Sta_full_name[i]\
                                +'\n'+str(int(Te_ref[3,top]))\
                                +'-'+str(int(Te_ref[4,top]))\
                                +'-'+str(int(Te_ref[5,top]))\
                                +'T'+str(hour)\
                                +':'+str(minute)\
                                +':'+str(second)+'LT',fontdict=Config['Font'])
                else:
                    plt.title(str(int(Te_ref[3,top]))\
                                +'-'+str(int(Te_ref[4,top]))\
                                +'-'+str(int(Te_ref[5,top]))\
                                +'T'+str(hour)\
                                +':'+str(minute)\
                                +':'+str(second)+'LT',fontdict=Config['Font'])
                plt
                for k in range(len(Format_keys)):
                    #ipdb.set_trace()
                    if Format_values[k]:
                        #fig.set_rasterized(True)
                        fig.savefig(folder+Format_keys[k]+'/'+sta_clu[i]\
                                    +'/Events/'+str(int(Te_ref[3,top]))\
                                    +'-'+str(int(Te_ref[4,top]))\
                                    +'-'+str(int(Te_ref[5,top]))\
                                    +'T'+str(hour)\
                                    +':'+str(minute)\
                                    +':'+str(second)+'LT.'+Format_keys[k])
                plt.close()

            acc_TBT = LA.norm(Te_TBT_t-Te_ISR_t, ord=2)/np.sqrt(index.shape)
            acc_Brace = LA.norm(Te_Brace_t-Te_ISR_t, ord=2)/np.sqrt(index.shape)
            acc_model = LA.norm(Te_model_t-Te_ISR_t, ord=2)/np.sqrt(index.shape)

            print('in ', sta_clu[i],', acc_TBT = ', acc_TBT)
            print('in ', sta_clu[i],', acc_Brace = ', acc_Brace)
            print('in ', sta_clu[i],', acc_model = ', acc_model)
            print('\r')

    def RO_predict(self,index_X):

        for i in range(len(Config['RO_predict']['Days'])):
            copyfile('Data/global_doy_'+str(Config['RO_predict']['Days'][i])+'_2009_range_10.mat',\
                     self.Project_path+'Data/Input/global_doy_'\
                     +str(Config['RO_predict']['Days'][i])+'_2009_range_10.mat')

        Days = Config['RO_predict']['Days']

        Coords = np.zeros([2,len(Stations)])
        Coords_geog = np.zeros([2,len(Stations)])
        Te_ranges = np.zeros([2,len(Stations)])

        for i in range(len(Stations)):
            Coords[:,i] = Config['RO_predict']['Coords'][Stations[i]]
            Coords_geog[:,i] = Config['RO_predict']['Coords_geog'][Stations[i]]
            Te_ranges[:,i] = Config['RO_predict']['Te'][Stations[i]]

        label = Config['RO_predict']['Months']

        if torch.cuda.is_available() and Config['NN_parameters']['GPU']:
            net = torch.load(self.net_path)
        else:
            net = torch.load(self.net_path, map_location='cpu')

        time_res = Config['RO_predict']['Time_res']
        alt_res = Config['RO_predict']['Alt_res']
        time_range = Config['RO_predict']['LT']
        alt_range = Config['RO_predict']['Alt']

        h_lim = Config['RO_predict']['H_lim']
        Coords_range = [value for value in Config['RO_predict']['Coord_range'].values()]


        #ipdb.set_trace()
        for i in range(len(Days)):
            try:
                f = h5py.File(self.Project_path+'Data/Input/global_doy_'\
                    +str(Config['RO_predict']['Days'][i])+'_2009_range_10.mat', 'r')
                Print_test("Keys: %s" % f.keys())
                a_group_key = list(f.keys())[0]
                #print(a_group_key)
                X_input = f[('X')]
                X_ref = f[('X_ref')]
            except:
                f = sio.loadmat(self.Project_path+'Data/Input/global_doy_'\
                    +str(Config['RO_predict']['Days'][i])+'_2009_range_10.mat')
                X_input = f['X'].T
                X_ref = f['X_ref'].T

            Print_test(X_input)
            TIE_file = 'tiegcm2.0_res5.0_'\
                                  +Config['RO_predict']['Months'][i]+ \
                                   '_smin_sech_001_test.nc'
            if not os.path.exists(self.Input_path+TIE_file):
                copyfile('Data/TIEGCM/'+TIE_file, self.Input_path+TIE_file)
            TIE_fid = Dataset(self.Input_path+TIE_file,'r')
            TIE_Lon, TIE_Lat, TIE_Alt, TIE_LT, TIE_Te, TIE_MI = ncread(TIE_fid)
            #import ipdb; ipdb.set_trace()
            TIE_TE, TIE_num = TIE_grid_h_time(TIE_Lon, TIE_Lat, TIE_Alt, TIE_LT, TIE_Te, TIE_MI, \
                             time_range, alt_range, time_res, alt_res, Coords_geog[:,i], [15,30], [100,5000])

            X_input[:,6] = X_input[:,6] * 1e-44

            X_input = X_input[:,index_X].T
            #print(X_input[5,:])
            #import ipdb; ipdb.set_trace()

            std_mean = sio.loadmat(self.Mean_std_path)
            stdX = std_mean['stdX']
            stdY = std_mean['stdY']
            meanX = std_mean['meanX']
            meanY = std_mean['meanY']

            X_input = torch.tensor((X_input - meanX.T) / stdX.T)

            if torch.cuda.is_available() and Config['NN_parameters']['GPU']:
                X_input = X_input.type(torch.FloatTensor).cuda().T
                out = net(X_input).cpu()
            else:
                X_input = X_input.type(torch.FloatTensor).T
                out = net(X_input)

            pred = out.detach().numpy()*stdY+meanY

            #将二维压缩为一维
            pred_y = pred.squeeze()
            Print_test(np.mean(pred_y))
            Print_test(np.min(pred_y))
            #X_ax = f[('X_ref')]
            Te = np.zeros([int((alt_range[1]-alt_range[0])/alt_res+1),\
                            int((time_range[1]-time_range[0])/time_res)+1]);
            axis = np.zeros([int((alt_range[1]-alt_range[0])/alt_res+1),\
                            int((time_range[1]-time_range[0])/time_res+1)]);
            num = np.zeros([int((alt_range[1]-alt_range[0])/alt_res+1),\
                            int((time_range[1]-time_range[0])/time_res+1)]);

            n = 1;
            m = 1;
            if torch.cuda.is_available() and Config['NN_parameters']['GPU']:
                X = X_input.cpu().numpy()*stdX+meanX
            else:
                X = X_input.numpy()*stdX+meanX

            #print(X.shape)
            #print(type(pred_y))
            #print(X_ref.shape)
            #print(X[:,6])
            for ij in range(len(Stations)):
                p = progressbar.ProgressBar()
                for j in p(range(time_range[0],time_range[1],time_res)):
                    for k in range(alt_range[0],alt_range[1]+alt_res,alt_res):
                        #print(X_ref[:,4])
                        #print(X_ref[:,5])
                        #ipdb.set_trace()
                        ind = np.where((X_ref[:,3]>=j) & (X_ref[:,3]<j+time_res) \
                                      &  (X_ref[:,0]==Config['RO_predict']['Year']) \
                                     & (pred_y>Config['Para_range']['Te'][0]) \
                                     & (pred_y<Config['Para_range']['Te'][1])\
                                       & (X_ref[:,6]>=k) & (X_ref[:,6]<k+alt_res) \
                                       & (np.abs(X_ref[:,4]-Coords[0, ij])<Coords_range[0]) \
                                       & (np.abs(X_ref[:,5]-Coords[1, ij])<Coords_range[1]) )
                        #print(type(pred_y[ind]))
                        #import ipdb; ipdb.set_trace()
                        if np.asarray(ind).shape[1]!=0:
                            Te[int((k-alt_range[0])/alt_res), int(j/time_res)] = LA.norm(pred_y[ind], ord=2)/np.sqrt(np.asarray(ind).shape[1])
                            num[int((k-alt_range[0])/alt_res), int(j/time_res)] = np.asarray(ind).shape[1]
                        else:
                            Te[int((k-alt_range[0])/alt_res), int(j/time_res)] = np.nan
                            num[int((k-alt_range[0])/alt_res), int(j/time_res)] = 0

                        Print_test(Te[int((k-alt_range[0])/alt_res), int(j/time_res)])
                        Print_test(num[int((k-alt_range[0])/alt_res), int(j/time_res)])
                        Print_test('alt = '+str(k)+', LT = '+str(j)+' in '+Stations[ij]+' at Day '+ str(Days[i]))

                Print_test(Stations[ij]+' at Day '+ str(Days[i]), True)

                #import ipdb; ipdb.set_trace()
                Te[:,-1] = Te[:,0]
                num[:,-1] = num[:,0]
                #creating sub-grid for quiver
                sub_grid_x = np.linspace(time_range[0],time_range[1],\
                            (time_range[1]-time_range[0])//time_res+1, dtype = np.int16)
                sub_grid_y = np.linspace(alt_range[0],alt_range[1],\
                            (alt_range[1]-alt_range[0])//alt_res+1, dtype = np.int16)+alt_res/2
                subx, suby = np.meshgrid(sub_grid_x, sub_grid_y)

                fig, (ax1,ax2,ax3) = plt.subplots(1,3, \
                    figsize=(Config['RO_predict']['Figs_size'][0],\
                    Config['RO_predict']['Figs_size'][1]))
                #im1 = ax1.contourf(mod_Ve_plane[xin:xfin,yin:yfin,0].T,60)
                #ax1.set_title('|Ve_plane|')
                boundary = np.linspace(Te_ranges[0,ij], Te_ranges[1,ij],\
                                    Config['RO_predict']['Density'])

                im1 = ax1.contourf(subx,suby,Te, 60, cmap=cm.jet,\
                                vmin=Te_ranges[0,ij],vmax=Te_ranges[1,ij])
                #im1.set_clim(Te_ranges[0,ij], Te_ranges[1,ij])
                fig.colorbar(im1,ax=ax1)
                #label_ticks = [str(ticks) for ticks in boundary ]
                #import ipdb; ipdb.set_trace()
                #im1.ax.set_xlim(Te_ranges[0,ij], Te_ranges[1,ij])
                #cbar.ax.set_yticklabels(label_ticks)
                #import ipdb; ipdb.set_trace()

                #im1.cma
                #im1 = ax1.contour(run.x[xin:xfin],run.y[yin:yfin],Psi[xin:xfin,yin:yfin].T,10,colors='red',linestyles='solid')
                ax1.set_title('Te outputs')


                #im2 = ax2.contourf(subx,suby,TIE_TE, 60, cmap=cm.jet)#
                #,vmin=Te_ranges[0,ij],vmax=Te_ranges[1,ij]
                #im1.set_clim(Te_ranges[0,ij], Te_ranges[1,ij])
                #fig.colorbar(im2,ax=ax2)
                #ax2.set_title('TIEGCM outputs')

                im3 = ax3.contourf(subx,suby,num,Config['RO_predict']['Density'])
                fig.colorbar(im3,ax=ax3)
                #im1 = ax1.contour(run.x[xin:xfin],run.y[yin:yfin],Psi[xin:xfin,yin:yfin].T,10,colors='red',linestyles='solid')
                ax3.set_title('Number')


                for j in range(len(Format_keys)):
                    if Format_values[j]:
                        if Format_keys[j]!='fig':
                            #fig.set_rasterized(True)
                            #import ipdb; ipdb.set_trace()
                            fig.savefig(self.Figs_path+Format_keys[j]+'/'+Stations[ij]\
                                        +'/Seasons/'+str(Days[i])\
                                        +'_'+str(Config['RO_predict']['Year'])\
                                        +'_'+str(Config['RO_predict']['Alt_res'])\
                                        +'_'+str(Config['RO_predict']['Time_res'])\
                                        +'.'+Format_keys[j])
                        else:
                            with open(self.Figs_path+Format_keys[j]+'/'+Stations[ij]\
                                        +'/Seasons/'+str(Days[i])\
                                        +'_'+str(Config['RO_predict']['Year'])\
                                        +'_'+str(Config['RO_predict']['Alt_res'])\
                                        +'_'+str(Config['RO_predict']['Time_res'])\
                                        +'.'+'.pickle', 'wb') as fig_file:

                                # Save to disk
                                pickle.dump(fig, fig_file)
                #ipdb.set_trace()
                plt.close()

#################################### IRI class #####################################
class Te_IRI_obj:
    """docstring for ."""
    def __init__(self, Ref):
        self.Ref = Ref
        #self.out = np.zeros([2,Ref.shape[1]])

    def Te_IRI_Read(self, ind):
        Ref = self.Ref
        start = time.time()
        year = int(Ref[3,ind])
        month = int(Ref[4,ind])
        day = int(Ref[5,ind])
        LT = int(Ref[6,ind])
        minute = int((Ref[6,ind]-LT)*60)
        second = int(((Ref[6,ind]-LT)*60-minute)*60)
        time_t = datetime(year, month, day, LT, minute, second)
        alt_km = [Ref[2,ind], Ref[2,ind], 100]
        lat = Ref[0,ind]
        lon = Ref[1,ind]

        iono_TBT = iri.IRI(time_t, alt_km, lat, lon, False)
        iono_Brace = iri.IRI(time_t, alt_km, lat, lon, True)
        end = time.time()
        Print_test('running time: '+str(end-start))
        #self.out[0,ind] = iono_TBT["Te"]
        #self.out[1,ind] = iono_Brace["Te"]
        #print('TBT result=', iono_TBT["Te"])
        #print('Brace result=', iono_Brace["Te"])
        #ipdb.set_trace()
        return iono_TBT["Te"], iono_Brace["Te"]
