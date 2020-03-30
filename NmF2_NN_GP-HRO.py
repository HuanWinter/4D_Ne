import tqdm
import torch
import gpytorch
from matplotlib import pyplot as plt
import scipy.io as sio
import numpy as np
import os
import torch.nn as nn
import datetime

smoke_test = ('CI' in os.environ)

# load the data
i = 0
data_RO = sio.loadmat('data/Delay/330w_0_minutes_geom.mat')

ALT = data_RO['out'][:, 0]
mLat = data_RO['out'][:, 1]

mLon = data_RO['out'][:, 2]
mLT = data_RO['out'][:, 3]
VTEC0 = data_RO['out'][:, 4]

varis = data_RO['out'][:, 5:9]
DoY = np.array(data_RO['out'][:, 9])
VTEC1 = data_RO['out'][:, 10]
NmF2 = data_RO['out'][:, 11]
year = data_RO['out'][:, 12]

Month = np.zeros(len(DoY))
DoM = np.zeros(len(DoY))

print(DoY.min())
print(DoY.max())

for i in range(len(DoY)):
    date = datetime.datetime(int(year[i]), 1, 1) +\
        datetime.timedelta(int(DoY[i]) - 1)

    Month[i] = date.month
    DoM[i] = date.day

print(DoM.min())
print(DoM.max())

LT = mLT + mLon/15

for i in range(len(LT)):
    if LT[i] >= 24:
        LT[i] -= 24
    elif LT[i] < 0:
        LT[i] += 24

Ref_iri = np.vstack([year, Month, DoM, LT, mLat, mLon]).T
print(Ref_iri.shape)

Vari_idx = [1, 2, 3, 5, 6, 7, 8, 9]
Target = [11]

ind = np.where((np.abs(mLat) > 0)
               & (~np.isnan(varis[:, 0]))
               & (~np.isnan(varis[:, 1]))
               & (~np.isnan(varis[:, 2]))
                   & (~np.isnan(varis[:,3])) \
                   #& (varis[:,3]<120) \
                   & (ALT>200) \
                   & (ALT<400) \
                   & (NmF2>np.exp(9.5)) \
                   & (NmF2<np.exp(15))
                   & (VTEC0 < 60)
                   & (VTEC0 > 0)
                  )[0]

def seed_torch(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(1029)

data = torch.from_numpy(data_RO['out'][ind,:]).type( torch.FloatTensor )
Ref_iri = Ref_iri[ind,:]
#X[:,2] = torch.from_numpy(LT[ind])

#Y = torch.from_numpy(np.sqrt(data['out'][ind,16]/(1.24e4))).type( torch.FloatTensor )

#ipdb.set_trace()

plt.hist(data[:,1],100)
plt.show()

#data[:,1] = np.cos(data[:,1]*np.pi/180)
#data[:,1] = np.sin(data[:,1]*np.pi/180)
#data[:,3] = np.sin(data[:,3]*np.pi/24)
#data[:,9] = np.sin(data[:,9]*np.pi/365)

ind_vtec = np.where((VTEC1[ind]>0) \
                     & (VTEC1[ind]<VTEC0[ind]/20))[0]
print(len(ind_vtec))
data[ind_vtec,4] = data[ind_vtec,4]+data[ind_vtec,10]
data[~ind_vtec,4] = data[~ind_vtec,4]*1.05

print(data[:,4].max())
print(data[:,4].min())

X = data[:,Vari_idx]
Y = data[:,Target]
Ref = data


X_save = X
Y_save = Y

#Y = (Y - Y_save.min())/(Y_save.max()-Y_save.min())

#ipdb.set_trace()
Ref = data[:,12]


### Normlisation
#X = X/LA.norm(X)
#Y = Y/LA.norm(Y)

print(X[1,:])

print(X.shape)
print(Y.shape)

idx_train = np.where((Ref!=2009) & (Ref!=2014))[0]
idx_test = np.where((Ref==2009) | (Ref==2014))[0]
idx_test1 = np.where((Ref==2009))[0]
idx_test2 = np.where((Ref==2014))[0]

#idx_test_t = idx_test2
#train_n = int(0.8 * len(X))
train_x = X[idx_train, :].contiguous()
train_y = Y[idx_train].contiguous()

test_x1 = X[idx_test1, :].contiguous()
test_y1 = Y[idx_test1].contiguous()
test_x2 = X[idx_test2, :].contiguous()
test_y2 = Y[idx_test2].contiguous()

print(np.abs(test_x1[:,0]).min())

'''
batch_size = 256
train_x = train_x[0:int(len(idx_train)/batch_size)*batch_size,:].view(
                [batch_size, int(len(idx_train)/batch_size),\
                len(Vari_idx)])
train_y = train_y[0:int(len(idx_train)/batch_size)*batch_size].view(
                [batch_size,int(len(idx_train)/batch_size)])

test_x = test_x[0:int(len(idx_test_t)/batch_size)*batch_size,:].view(
                [batch_size, int(len(idx_test_t)/batch_size),\
                len(Vari_idx)])
test_y = test_y[0:int(len(idx_test_t)/batch_size)*batch_size].view(
                [batch_size,int(len(idx_test_t)/batch_size)])

if torch.cuda.is_available():
    test_x1, test_y1, test_x2, test_y2 =\
    test_x1.cuda(), test_y1.cuda(),\
    test_x2.cuda(), test_y2.cuda(),
'''
print(train_x.shape)
print(train_y.shape)

print(test_x1.shape)
print(test_x2.shape)
#batch_num = torch.Size([train_x.shape[0]])
#print(batch_num)

############################ iono data #########################
import scipy.io as sio
from multiprocessing import cpu_count, Pool
from random import sample
import tqdm

data = sio.loadmat('../Omni_data/HRO.mat')


Ref = data['Ref']

ind_2009 = np.where((Ref[:,0]==2009))[0]

ind_2002 = np.where((Ref[:,0]==2014))[0]


X_iono_2002, Y_iono_2002 = torch.from_numpy(data['X'][ind_2002,:]).type(torch.FloatTensor),\
                torch.from_numpy(data['Y'][ind_2002,:]).type(torch.FloatTensor)

X_iono_2009, Y_iono_2009 = torch.from_numpy(data['X'][ind_2009,:]).type(torch.FloatTensor),\
                torch.from_numpy(data['Y'][ind_2009,:]).type(torch.FloatTensor)

cores = cpu_count()
print(cores)
#X_save = X_train
#Y_save = Y_train
#print((Y_save.max() - Y_save.min()))

#X_train = (X_train - X_train.min())/(X_train.max() - X_train.min())
#Y_train = (Y_train - Y_train.min())/(Y_train.max() - Y_train.min())

X_iono = X_iono_2002
Y_iono = Y_iono_2002

print(X_iono.shape)
print(Y_iono.shape)
#X_iono, Y_iono = X_iono.cuda(), Y_iono.cuda()

######################################### models ######################

from skorch import NeuralNetRegressor
from skorch.callbacks import ProgressBar, Checkpoint
from torch.nn import ReLU, Linear, Tanh, Sigmoid, LeakyReLU

data_dim = train_x.size(-1)

class MyModule(nn.Module):
    def __init__(self, num_units=10):
        super(MyModule, self).__init__()

        self.dense1 = Linear(data_dim, 32)
        self.dense2 = Linear(32, 16)
        self.dense3 = Linear(16, 8)
        self.nonlin = ReLU()
        self.output = nn.Linear(8, 1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense1(X))
        X = self.nonlin(self.dense2(X))
        X = self.nonlin(self.dense3(X))
        X = self.output(X)
        return X

my_callbacks = [Checkpoint()]
if True:
    my_callbacks.append(ProgressBar())

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', nn.Linear(data_dim, 100))
        self.add_module('relu1', nn.ReLU())
        self.add_module('linear2', nn.Linear(100, 50))
        self.add_module('relu2', nn.ReLU())
        self.add_module('linear3', nn.Linear(50, 25))
        self.add_module('relu3', nn.ReLU())
        self.add_module('linear4', nn.Linear(25, 2))

feature_extractor = LargeFeatureExtractor()

class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                num_dims=2, grid_size=100
            )
            self.feature_extractor = feature_extractor

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            # We're also scaling the features so that they're nice values
            projected_x = self.feature_extractor(x)
            projected_x = projected_x - projected_x.min(0)[0]
            projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


################################## train_GP function ##########################

training_iterations = 2 if smoke_test else 100

def train_GP(train_x, train_y, 
             test_x_high, test_y_high, 
             test_x_low, test_y_low, 
             test_x_iono, test_y_iono, 
             model, likelihood,optimizer,
             save_name):
    
    seed_torch(1029)
    
    # Find optimal model hyperparameters
    out_test = np.zeros(training_iterations)
    out_test_low = np.zeros(training_iterations)
    out_test_high = np.zeros(training_iterations)
    out_test_iono = np.zeros(training_iterations)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood, model)
    iterator = tqdm.tqdm_notebook(range(training_iterations))
    
    for i in iterator:
                
        model.train()
        likelihood.train()

        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        #ipdb.set_trace()
        
        output = model(train_x)
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y)
        #ipdb.set_trace()
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()
        
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            #ipdb.set_trace()
            preds_high = model(test_x_high)
            preds_low = model(test_x_low)
            preds_iono = model(test_x_iono)
        
        out_test_low[i] = torch.sqrt(torch.mean((preds_low.mean - test_y_low)**2))
        out_test_high[i] = torch.sqrt(torch.mean((preds_high.mean - test_y_high)**2))
        out_test_iono[i] = torch.sqrt(torch.mean((preds_iono.mean - test_y_iono)**2))
        out_test[i] = torch.sqrt(torch.mean((preds_low.mean - test_y_low)**2)) \
                          +torch.sqrt(torch.mean((preds_high.mean - test_y_high)**2)) 
        out_test[i] = out_test_iono[i] # check a different criteria (iono)
        out_test_diff = np.diff(out_test) 
        
        if i>1: 
            if out_test[0:i].min() > out_test[i]:
                state_dict = model.state_dict()
                likelihood_state_dict = likelihood.state_dict()
                torch.save({'model': state_dict,\
                            'likelihood': likelihood_state_dict},\
                             save_name)
        #print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))
        #print('Test RMSE: {}'.format(out_test[i]))
        
        if i>5 & (out_test_diff[i-5:i]>=0).sum()==5:
            break
    #ipdb.set_trace()
    return out_test_low[0:i].min(), out_test_high[0:i].min(), out_test_iono[0:i].min()
        
###################################### main model ########################

import pickle
from progressbar import progressbar
y_save = train_y
#Train_y = (train_y - train_y.min())/(train_y.max() - train_y.min())
Train_y = np.array(train_y)
print(train_x.shape)
print(train_y.shape)
print(train_y.max())
print(train_y.min())

def norm_ah(X):
    num = X.shape[1]
    out = np.zeros(X.shape)
    for i in range(num):
        dis = X[:,i].max() - X[:,i].min()
        out[:,i] = (X[:,i] - X[:,i].min())/dis
        
    #ipdb.set_trace()
    
    return torch.from_numpy(out).type(torch.FloatTensor)

lat_interval = 30
t_interval = 0.5
buffer = 5
lat_range = range(-90,90,lat_interval)
t_range = range(0,int(1/t_interval))
p = progressbar

RMSE_2009_NN = np.zeros([int(180/lat_interval),int(1/t_interval)])
RMSE_2014_NN = np.zeros([int(180/lat_interval),int(1/t_interval)])
RMSE_iono_NN = np.zeros([int(180/lat_interval),int(1/t_interval)])
RMSE_2009_GP = np.zeros([int(180/lat_interval),int(1/t_interval)])
RMSE_2014_GP = np.zeros([int(180/lat_interval),int(1/t_interval)])
RMSE_iono_GP = np.zeros([int(180/lat_interval),int(1/t_interval)])
Re_2009_NN = np.zeros([int(180/lat_interval),int(1/t_interval)])
Re_2014_NN = np.zeros([int(180/lat_interval),int(1/t_interval)])

#seed_torch(1029)
net = MyModule()
train_x_save = np.array(train_x)
test_x1_save = np.array(test_x1)
test_x2_save = np.array(test_x2)
test_x_iono_save = np.array(X_iono)

## normlisation
train_x = norm_ah(train_x)
#train_y = norm_ah(train_y)
test_x1 = norm_ah(test_x1)
#test_y1 = norm_ah(test_y1)
test_x2 = norm_ah(test_x2)
#test_y2 = norm_ah(test_y2)
test_x_iono = norm_ah(X_iono)
test_y_iono = Y_iono

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

for lat_sta in p(lat_range):
    
    for LT_sta in t_range:
        
        net.apply(init_weights)
        Model = NeuralNetRegressor(
            net,
            max_epochs=100,
            lr=0.02,
            #train_split=None,
            batch_size=256,
            #callbacks=[ProgressBar()],
            callbacks=my_callbacks,
            optimizer=torch.optim.AdamW,
            device='cuda',
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
        )

        idx_train = np.where((train_x_save[:,0]>lat_sta-buffer) \
                      & (train_x_save[:,0]<=lat_sta+lat_interval+buffer)\
                      & (train_x_save[:,2]>LT_sta*t_interval) \
                      & (train_x_save[:,2]<=(LT_sta+1)*t_interval)
                            )[0]
        idx_test1 = np.where((test_x1_save[:,0]>lat_sta) \
                      & (test_x1_save[:,0]<=lat_sta+lat_interval)\
                      & (test_x1_save[:,2]>LT_sta*t_interval) \
                      & (test_x1_save[:,2]<=(LT_sta+1)*t_interval)
                            )[0]
        idx_test2 = np.where((test_x2_save[:,0]>lat_sta) \
                      & (test_x2_save[:,0]<=lat_sta+lat_interval)\
                      & (test_x2_save[:,2]>LT_sta*t_interval) \
                      & (test_x2_save[:,2]<=(LT_sta+1)*t_interval)
                            )[0]
        idx_test_iono = np.where((test_x_iono_save[:,0]>lat_sta) \
                      & (test_x_iono_save[:,0]<=lat_sta+lat_interval)\
                      & (test_x_iono_save[:,2]>LT_sta*t_interval) \
                      & (test_x_iono_save[:,2]<=(LT_sta+1)*t_interval)
                            )[0]
        #ipdb.set_trace()
        
        train_x_GP = train_x[idx_train].cuda()
        train_y_GP = torch.from_numpy(Train_y[idx_train].squeeze()).cuda()
        test_x1_GP = test_x1[idx_test1].cuda()
        test_y1_GP = test_y1[idx_test1].cuda()
        test_x2_GP = test_x2[idx_test2].cuda()
        test_y2_GP = test_y2[idx_test2].cuda()
        test_x_iono_GP = test_x_iono[idx_test_iono].cuda()
        test_y_iono_GP = test_y_iono[idx_test_iono].cuda()
        
        seed_torch(1029)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model_GP = GPRegressionModel(train_x_GP,
                                     train_y_GP,
                                     likelihood)
        #ipdb.set_trace()
        # Use the adam optimizer
        optimizer = torch.optim.AdamW([
            {'params': model_GP.feature_extractor.parameters()},
            {'params': model_GP.covar_module.parameters()},
            {'params': model_GP.mean_module.parameters()},
            {'params': model_GP.likelihood.parameters()},
        ], lr=0.02)


        if torch.cuda.is_available():
            model_GP = model_GP.cuda()
            likelihood = likelihood.cuda()
        
        #ipdb.set_trace()
        model_GP_name = 'NmF2_Model/GP_'\
                +str(lat_sta)\
                +'_'\
                +str(int(LT_sta*24*t_interval))\
                +'.dat'
        
        RMSE_2009_GP_t, RMSE_2014_GP_t, RMSE_iono_GP_t = train_GP(train_x=train_x_GP, 
                   train_y=train_y_GP, 
                   model=model_GP, 
                   likelihood=likelihood,
                   iteration=iterator, 
                   optimizer=optimizer,
                   test_x_low=test_x1_GP,
                   test_y_low=test_y1_GP,
                   test_x_high=test_x2_GP,
                   test_y_high=test_y2_GP,
                   test_x_iono=test_x_iono_GP,
                   test_y_iono=test_y_iono_GP,
                   save_name=model_GP_name)
        
        RMSE_2009_GP[int((lat_sta+90)/lat_interval), LT_sta] = RMSE_2009_GP_t
        RMSE_2014_GP[int((lat_sta+90)/lat_interval), LT_sta] = RMSE_2014_GP_t
        RMSE_iono_GP[int((lat_sta+90)/lat_interval), LT_sta] = RMSE_iono_GP_t

        Model.fit(train_x[idx_train],Train_y[idx_train].reshape(-1, 1))

        model_name = 'NmF2_Model/'+str(lat_sta)+'_'+str(int(LT_sta*24*t_interval))+'.pkl'

        # Load best parameters
        if True:
            Model.load_params(f_params='params.pt')

        Y_pred1 = torch.from_numpy(Model.predict(test_x1[idx_test1]))
        Y_pred2 = torch.from_numpy(Model.predict(test_x2[idx_test2]))
        Y_pred_iono = torch.from_numpy(Model.predict(test_x_iono[idx_test_iono]))
        RMSE_2009_NN[int((lat_sta+90)/lat_interval), LT_sta]\
        = torch.sqrt(torch.mean((Y_pred1.squeeze() - test_y1[idx_test1].cpu())**2))
        RMSE_2014_NN[int((lat_sta+90)/lat_interval), LT_sta]\
        = torch.sqrt(torch.mean((Y_pred2.squeeze() - test_y2[idx_test2].cpu())**2))
        RMSE_iono_NN[int((lat_sta+90)/lat_interval), LT_sta]\
        = torch.sqrt(torch.mean((Y_pred_iono.squeeze() - test_y_iono[idx_test_iono].cpu())**2))
        Re_2009_NN[int((lat_sta+90)/lat_interval), LT_sta]\
        = torch.median(torch.abs(Y_pred1.squeeze() - test_y1[idx_test1].cpu())/test_y1[idx_test1].cpu())
        Re_2014_NN[int((lat_sta+90)/lat_interval), LT_sta]\
        = torch.median(torch.abs(Y_pred2.squeeze() - test_y2[idx_test2].cpu())/test_y2[idx_test2].cpu())
        print('near', lat_sta, ' Lat by', LT_sta*24*t_interval)
        print('Using simple NN, Test RMSE in',
              '2009: {}'.format(RMSE_2009_NN[int((lat_sta+90)/lat_interval), LT_sta]))
        print('Using simple NN, Test RMSE in',
              '2014: {}'.format(RMSE_2014_NN[int((lat_sta+90)/lat_interval), LT_sta]))
        print('Using simple NN, Test RMSE in',
              'iono: {}'.format(RMSE_iono_NN[int((lat_sta+90)/lat_interval), LT_sta]))
        print('Using KISS-GP, Test RMSE in',
              '2009: {}'.format(RMSE_2009_GP[int((lat_sta+90)/lat_interval), LT_sta]))
        print('Using KISS-GP, Test RMSE in',
              '2014: {}'.format(RMSE_2014_GP[int((lat_sta+90)/lat_interval), LT_sta]))
        print('Using KISS-GP, Test RMSE in',
              'iono: {}'.format(RMSE_iono_GP[int((lat_sta+90)/lat_interval), LT_sta]))
        # saving
        with open(model_name, 'wb') as f:
            pickle.dump(net, f)

############################# results ###########################

print(RMSE_2009_NN)
print(RMSE_2009_GP)
print(RMSE_iono_GP)
print(RMSE_iono_NN)
print('NN:')
print('2009:',RMSE_2009_NN.reshape(1,-1).mean())
print('2014:',RMSE_2014_NN.reshape(1,-1).mean())
print('iono:',RMSE_iono_NN.reshape(1,-1).mean())
print('GP:')
print('2009:',RMSE_2009_GP.reshape(1,-1).mean())
print('2014:',RMSE_2014_GP.reshape(1,-1).mean())
print('iono:',RMSE_iono_GP.reshape(1,-1).mean())
print((RMSE_iono_NN - RMSE_iono_GP)/RMSE_iono_NN)
