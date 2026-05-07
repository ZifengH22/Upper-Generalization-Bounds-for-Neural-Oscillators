import torch 
import time 
import numpy as np
from tqdm import tqdm 
from torch.utils.data import DataLoader  
from sklearn.model_selection import train_test_split 

import os 
import shutil 
import random 
import sys 
import contextlib 
import scipy.io   
import matplotlib.pyplot as plt  
# from Network_without_GRU import myDNN, myRK4GRUcell
from Network_without_GRU_ResNet_simple_norm_RK2_simple_withoutbatchnorm import topDNN, myRK4GRUcell
import h5py 
import copy

seed = 1328
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
class Args: 
    def __init__(self) -> None:  
        
        self.dt = 0.01                             
        self.SV_feature = 2                        # space state  [x  xdot]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layers = [20,10] 
        self.top_layers = [self.layers[-1],15,15,1]
        
        # Epoch
        self.epochs = 3000
        self.time_length = 2500
        # file path
        self.data_path = r'data/' 
        self.model_data_path = 'Data_path_time_length_'+str(self.time_length)+'/data/'
        self.modelsave_path = 'Data_path_time_length_'+str(self.time_length)+'/Results/'
        if not os.path.exists(self.modelsave_path):
            os.makedirs(self.modelsave_path)

def load_matdata(args,seed):
    data_path_F = args.model_data_path + 'F_train_'+str(seed)+'.mat'
    data_path_X_dX_input_train = args.model_data_path + 'X_dX_input_train.mat'
    data_path_X_dX_output_train = args.model_data_path + 'X_dX_output_train_'+str(seed)+'.mat'
    data_path_E_X_output_train = args.model_data_path + 'E_X_output_train_'+str(seed)+'.mat'
    data_path_t_train = args.model_data_path + 't_train.mat'

    data_path_Acc = args.data_path + 'Acc.mat'
    data_path_E_X = args.data_path + 'E_X1_response.mat'
    
    ##training data    
    with h5py.File(data_path_F, 'r') as file:
        variable_name = list(file.keys())[0]
        F_input_train = file[variable_name][:]
    F_input_train = F_input_train[:, np.newaxis]
    F_input_train = np.transpose(F_input_train, axes = [2,0,1])
    
    with h5py.File(data_path_X_dX_input_train, 'r') as file:
        variable_name = list(file.keys())[0]
        X_dX_input_train = file[variable_name][:]
    X_dX_input_train = np.transpose(X_dX_input_train, axes = [2,1,0])

    with h5py.File(data_path_X_dX_output_train, 'r') as file:
        variable_name = list(file.keys())[0]
        X_dX_output_train = file[variable_name][:]
    X_dX_output_train = X_dX_output_train[:, np.newaxis]
    X_dX_output_train = np.transpose(X_dX_output_train, axes = [2,0,1])
    
    with h5py.File(data_path_E_X_output_train, 'r') as file:
        variable_name = list(file.keys())[0]
        E_X_output_train = file[variable_name][:]
    E_X_output_train = E_X_output_train[:, np.newaxis]
    E_X_output_train = np.transpose(E_X_output_train, axes = [2,0,1])
    
    with h5py.File(data_path_t_train, 'r') as file:
        variable_name = list(file.keys())[0]
        t_train = file[variable_name][:]
    t_train = t_train[:, np.newaxis]
    t_train = np.transpose(t_train, axes = [2,0,1])

    ##test data
    with h5py.File(data_path_Acc, 'r') as file:
        variable_name = list(file.keys())[0]
        Acc_test = file[variable_name][:]
    Acc_test = Acc_test[:, np.newaxis] 
    Acc_test = np.transpose(Acc_test, axes = [0,2,1])
    Acc_test = Acc_test[:,0:int(args.time_length),:]
    
    # with h5py.File(data_path_X, 'r') as file:
    #     variable_name = list(file.keys())[0]
    #     X_test = file[variable_name][:]
    # X_test = X_test[:, np.newaxis]
    # X_test = np.transpose(X_test, axes = [0,2,1])
    # X_test = X_test[:,0:int(args.time_length),:]
    X_test = 0.

    with h5py.File(data_path_E_X, 'r') as file:
        variable_name = list(file.keys())[0]
        E_X_test = file[variable_name][:]
    E_X_test = E_X_test[:, np.newaxis]
    E_X_test = np.transpose(E_X_test, axes = [0,2,1])
    E_X_test = E_X_test[:,0:int(args.time_length),:]

    return F_input_train, X_dX_input_train, X_dX_output_train, E_X_output_train, t_train, Acc_test, X_test, E_X_test

####################################
# test model
args = Args()
_, _, _, _, t_train, Acc_test, _, E_X_test = load_matdata(args,seed)

# F_input_train = torch.from_numpy(F_input_train)
# X_dX_input_train = torch.from_numpy(X_dX_input_train)
# E_X_output_train = torch.from_numpy(E_X_output_train)
t_train = torch.from_numpy(t_train)

# del Acc_test, X_test, E_X_test 
# test_exc = F_input_train
# test_x_xdot = E_X_output_train
# test_initial = X_dX_input_train
# test_t = t_train
# num_sample_x_dx = test_x_xdot.shape[0]

Acc_test = torch.from_numpy(Acc_test)
test_exc = Acc_test
test_x_xdot = torch.from_numpy(E_X_test)
test_initial = torch.zeros(test_exc.shape[0],1,test_exc.shape[2]*2)
test_t = t_train
num_sample_x_dx = test_x_xdot.shape[0]
del Acc_test, E_X_test

###
num_input_layer_X_dX = torch.numel(test_initial[0,:,:])
num_input_layer_F = len(test_exc[0,0,:])
args.layers.insert(0,num_input_layer_F + num_input_layer_X_dX*args.layers[-1])
args.top_layers[0] += (test_t.shape[2] + test_exc.shape[2])

gru_step = torch.numel(test_exc[0,:,0])
step_delay_F = 0
step_delay_X_dX = 0

########
# load model
modelsave_path = args.modelsave_path
data_path = args.data_path

RK4GRUcell = myRK4GRUcell(args).to(args.device)
top_DNN = topDNN(args.top_layers, lastbias = True).to(args.device)

num_sample = scipy.io.loadmat(args.model_data_path + 'Sample_size.mat')
str_layers = '_'.join(map(str, args.layers))
str_top_layers = '_'.join(map(str, args.top_layers))

# path_save_RK4GRUcell = modelsave_path + 'RK4GRUcell_best_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '_' + str(seed) + '.pth'
# path_save_topDNN = modelsave_path + 'topDNN_best_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '_' + str(seed) + '.pth'

# path_save_RK4GRUcell = modelsave_path + 'RK4GRUcell_last_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '_' + str(seed) + '.pth'
# path_save_topDNN = modelsave_path + 'topDNN_last_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '_' + str(seed) + '.pth'

# path_save_RK4GRUcell = modelsave_path + 'RK4GRUcell_smallest_1_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '_' + str(seed) + '.pth'
# path_save_topDNN = modelsave_path + 'topDNN_smallest_1_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '_' + str(seed) + '.pth'

path_save_RK4GRUcell = modelsave_path + 'RK4GRUcell_smallest_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '_' + str(seed) + '.pth'
path_save_topDNN = modelsave_path + 'topDNN_smallest_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '_' + str(seed) + '.pth'

print(path_save_RK4GRUcell)
print(path_save_topDNN)

RK4GRUcell.load_state_dict(torch.load(path_save_RK4GRUcell, map_location=torch.device('cpu'),weights_only=True))
top_DNN.load_state_dict(torch.load(path_save_topDNN, map_location=torch.device('cpu'),weights_only=True))

###############model parameters##################
# RK4GRUcell_copy = copy.deepcopy(RK4GRUcell.state_dict())
# top_DNN_copy = copy.deepcopy(top_DNN.state_dict())

# for name, param in RK4GRUcell_copy.items():
#     print(name, param.shape)
#     if '_layer' in name and 'bias' in name:
#         para_data = param.detach().cpu().clone().numpy()
#         para_name = 'B_RK4GRUcell'+name[21]
#         file_path = args.model_data_path+para_name+'.mat'
#         scipy.io.savemat(file_path, {para_name: para_data})

#         print(np.sum(para_data**2))
#         print(np.sum(np.abs((para_data))))
#         print(np.max(np.abs((para_data))))
        
#     if '_layer' in name and 'weight' in name:
#         para_data = param.detach().cpu().clone().numpy()
#         para_name = 'W_RK4GRUcell'+name[21]
#         file_path = args.model_data_path+para_name+'.mat'
#         scipy.io.savemat(file_path, {para_name: para_data})

#         print(np.sum(para_data**2))
#         print(np.sum(np.abs((para_data))))
#         print(np.max(np.abs((para_data))))
    

# for name, param in top_DNN_copy.items():
#     print(name, param.shape)
#     if '_layer' in name and 'bias' in name:
#         para_data = param.detach().cpu().clone().numpy()
#         para_name = 'B_top_DNN'+name[17]
#         file_path = args.model_data_path+para_name+'.mat'
#         scipy.io.savemat(file_path, {para_name: para_data})

#         print(np.sum(para_data**2))
#         print(np.sum(np.abs((para_data))))
#         print(np.max(np.abs((para_data))))
        
#     if '_layer' in name and 'weight' in name:
#         para_data = param.detach().cpu().clone().numpy()
#         para_name = 'W_top_DNN'+name[17]
#         file_path = args.model_data_path+para_name+'.mat'
#         scipy.io.savemat(file_path, {para_name: para_data})

#         print(np.sum(para_data**2))
#         print(np.sum(np.abs((para_data))))
#         print(np.max(np.abs((para_data))))
#################################################

#########calculate pred_response###########
RK4GRUcell.eval()
top_DNN.eval()

num_sample_once = 200
iter_num = round(num_sample_x_dx/num_sample_once)
pred_state_final = torch.zeros_like(test_x_xdot).to(torch.float32)
test_exc = test_exc.to(torch.float32).to(args.device)

for iter_id in range(iter_num):
    print(f'iter_id = {iter_id}')
    
    test_initial_temp = test_initial[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,:,:].to(torch.float32).to(args.device)
    pred_state = torch.zeros_like(test_x_xdot[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,:,:]).to(torch.float32).to(args.device)  # (1,1499,3) # [x xdot g] 
    SVi_delay_temp = torch.cat( (test_initial_temp[:,:,0::2].repeat(1, 1,args.layers[-1]), test_initial_temp[:,:,1::2].repeat(1, 1,args.layers[-1]) ),-1)        
    T_time = test_t[0:num_sample_once,:,:].to(torch.float32).to(args.device)

    svj = SVi_delay_temp
    top_DNN_input = torch.cat((svj[:,:,:svj.shape[2]//2],T_time[:,0:1,:],test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,0:1,:]),-1)
    pred_state[:,0:1,:] = top_DNN(top_DNN_input)
    
    for i in tqdm(range(pred_state.shape[1] - step_delay_X_dX - 1), desc='Predict tracks'):
        exci_delay = test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,i:(1 + i), :]
        excj = test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,(i + 1):(i + 2), :]

        svj,_,_,_,_ = RK4GRUcell(SVi_delay_temp,step_delay_X_dX, step_delay_F,exci_delay,excj)
        top_DNN_input = torch.cat((svj[:,:,:svj.shape[2]//2],T_time[:,(i + 1):(i + 2),:],test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,0:1,:]),-1)
        pred_state[:,i+1:i+2,:] = top_DNN(top_DNN_input)
        SVi_delay_temp = svj
   
    pred_state_final[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,:,:] = pred_state.detach()

####saving data#####
E_X_pred = np.transpose(pred_state_final[:,:,0].numpy(), axes = [1,0])
E_X_pred_dict = {'E_X_pred_'+str(args.time_length) + '_' + str(seed):E_X_pred} 
scipy.io.savemat(args.data_path + 'E_X_pred_'+str(args.time_length)+ '_' + str(seed)+'.mat', E_X_pred_dict) 

##################plot#####################
test_x_xdot.shape
pred_state_final.shape

dt = args.dt
t = np.linspace(0, pred_state_final.shape[1]-1, pred_state_final.shape[1])*dt
index = 3456
plt.figure(1)
plt.plot(t,test_x_xdot[index,:,0],linestyle = '-',color = 'k')
plt.plot(t,pred_state_final[index,:,0],linestyle = '--',color = 'r')
plt.show()

plt.figure(2)
plt.plot(test_exc[index,:,0],linestyle = '-',color = 'k')
plt.show()

weight = torch.ones_like(test_x_xdot)
weight = torch.square(weight)
error_X_pred_all = (test_x_xdot - pred_state_final);
error_X_mean_all = torch.mean((error_X_pred_all*weight)**2)
error_X_mean_all_relative = error_X_mean_all/torch.mean((test_x_xdot)**2)
print(error_X_mean_all)
print(error_X_mean_all_relative)

error_X_pred = (test_x_xdot[index,:,0] - pred_state_final[index,:,0]);
error_X_mean = torch.mean((error_X_pred*weight[index,:,:])**2)
print(error_X_mean)

plt.figure(3)
plt.plot(t,error_X_pred,linestyle = '--',color = 'r')
plt.show()