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
from Network_without_GRU_ResNet_simple_norm_RK2_simple_withoutbatchnorm import topDNN, myRK4GRUcell
import timeit
import h5py
import copy

seed = 1228
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
###############################################################################
class Args: 
    def __init__(self) -> None: 
        self.dt = 0.01                             #self.dt dt = 0.01
        self.SV_feature = 2                        # space state  [x  xdot]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layers = [40,10] 
        self.top_layers = [self.layers[-1],20,1]
        
        ### training parameters
        self.batch_size = 100  #  batch size 100
        self.seq_len = 1  
        self.size_all = 3200
        self.lamda_u = 0.003
        self.p_loss_2 = 1   
        
        # Learning rate
        self.lr = 0.01
        self.lr_step = 100
        self.lr_gamma = 0.965
       
        # Epoch
        self.epochs = 1000
        self.valper = 1
        
        # file path
        self.data_path = r'data/' 
        self.modelsave_path = r'Results/'
        if not os.path.exists(self.modelsave_path):
            os.makedirs(self.modelsave_path)

def load_matdata_train(args):
    # data loading
    data_path_F = args.data_path + 'F_train.mat'
    data_path_X_dX_input_train = args.data_path + 'X_dX_input_train.mat'
    data_path_X_dX_output_train = args.data_path + 'X_dX_output_train.mat'
    data_path_E_X_output_train = args.data_path + 'E_X_output_train.mat'
    data_path_t_train = args.data_path + 't_train.mat'
    
    data_path_coef_F = args.data_path + 'coef_F.mat'
    data_path_coef_X = args.data_path + 'coef_X_output.mat'
    data_path_coef_dX = args.data_path + 'coef_dX_output.mat'
    data_path_coef_ddX = args.data_path + 'coef_ddX_output.mat'
    data_path_coef_E_X = args.data_path + 'coef_E_X_output.mat'
    
    with h5py.File(data_path_F, 'r') as file:
        variable_name = list(file.keys())[0]
        F_input_train = file[variable_name][:]
    F_input_train = F_input_train[:, np.newaxis]
    F_input_train = np.transpose(F_input_train, axes = [2,0,1])
    
    with h5py.File(data_path_X_dX_input_train, 'r') as file:
        variable_name = list(file.keys())[0]
        X_dX_input_train = file[variable_name][:]
    # X_dX_input_train = X_dX_input_train[:, np.newaxis] 
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
   
    coef_F = scipy.io.loadmat(data_path_coef_F)
    coef_F = coef_F['coef_F'][0,0]

    coef_X = scipy.io.loadmat(data_path_coef_X)
    coef_X = coef_X['coef_X_output'][0,0]

    coef_dX = scipy.io.loadmat(data_path_coef_dX)
    coef_dX = coef_dX['coef_dX_output'][0,0]
    
    coef_ddX = scipy.io.loadmat(data_path_coef_ddX)
    coef_ddX = coef_ddX['coef_ddX_output'][0,0]

    coef_E_X = scipy.io.loadmat(data_path_coef_E_X)
    coef_E_X = coef_E_X['coef_E_X_output'][0,0]
    
    return F_input_train, X_dX_input_train, X_dX_output_train, E_X_output_train, t_train, coef_F,coef_X,coef_dX,coef_ddX, coef_E_X

###############################################################################
# main function
def train_RK4PIGRU_main(args):
    modelsave_path = args.modelsave_path 
    data_path = args.data_path 
    
    F_input, X_dX_input, X_dX_output, E_X_output, t_input, coef_F,coef_X,coef_dX,coef_ddX,coef_E_X = load_matdata_train(args) #[number,length,feature]
    num_sample = scipy.io.loadmat(args.data_path + 'Sample_size.mat')
    num_copy_training_sample = int(args.size_all/int(num_sample[list(num_sample.keys())[-1]].squeeze()))
    sample_size = int(num_sample[list(num_sample.keys())[-1]].squeeze())

    #################
    indices = np.arange(X_dX_input.shape[0])
    # np.random.shuffle(indices)
    
    train_indices = indices[0:round(indices.shape[0]*0.8)]
    val_indices = indices[round(indices.shape[0]*0.8):indices.shape[0]]
    
    F_input_train = torch.from_numpy(F_input[train_indices, :, :])
    X_dX_input_train = torch.from_numpy(X_dX_input[train_indices, :, :])
    X_dX_output_train = torch.from_numpy(X_dX_output[train_indices, :, :])
    E_X_output_train = torch.from_numpy(E_X_output[train_indices, :, :])
    t_input_train = torch.from_numpy(t_input[train_indices, :, :])

    F_input_val = torch.from_numpy(F_input[val_indices, :, :])
    X_dX_input_val = torch.from_numpy(X_dX_input[val_indices, :, :])
    X_dX_output_val = torch.from_numpy(X_dX_output[val_indices, :, :])
    E_X_output_val = torch.from_numpy(E_X_output[val_indices, :, :])
    t_input_val = torch.from_numpy(t_input[val_indices, :, :])

    ########sample copy########
    F_input_train = F_input_train.repeat(num_copy_training_sample,1,1)
    X_dX_input_train = X_dX_input_train.repeat(num_copy_training_sample,1,1)
    X_dX_output_train = X_dX_output_train.repeat(num_copy_training_sample,1,1)
    E_X_output_train = E_X_output_train.repeat(num_copy_training_sample,1,1)
    t_input_train = t_input_train.repeat(num_copy_training_sample,1,1)

    F_input_val = F_input_val.repeat(num_copy_training_sample,1,1)
    X_dX_input_val = X_dX_input_val.repeat(num_copy_training_sample,1,1)
    X_dX_output_val = X_dX_output_val.repeat(num_copy_training_sample,1,1)
    E_X_output_val = E_X_output_val.repeat(num_copy_training_sample,1,1)
    t_input_val = t_input_val.repeat(num_copy_training_sample,1,1)

    #####################
    train_dataset = torch.utils.data.TensorDataset(F_input_train, X_dX_input_train,X_dX_output_train,t_input_train)
    val_dataset = torch.utils.data.TensorDataset(F_input_val, X_dX_input_val, X_dX_output_val,t_input_val)
     
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = False)
    val_dataloader = DataLoader(dataset = val_dataset, batch_size = args.batch_size, shuffle = False)
    
    ####layers,gru_step, F, X, dX, delay####
    num_input_layer_X_dX = torch.numel(X_dX_input_train[0,:,:])
    num_input_layer_F = (torch.numel(F_input_train[0,:,0]) - torch.numel(X_dX_output_train[0,:,0]) + 1)*len(F_input_train[0,0,:])
    args.layers.insert(0,num_input_layer_F + num_input_layer_X_dX*args.layers[-1])
    args.top_layers[0] += (t_input.shape[2] + F_input_train.shape[2])

    gru_step = torch.numel(X_dX_output_train[0,:,0])
    step_delay_F = torch.numel(F_input_train[0,:,0]) - torch.numel(X_dX_output_train[0,:,0])
    step_delay_X_dX = torch.numel(X_dX_input_train[0,:,0]) - 1

    ####neural network and optimization method and parameters
    RK4GRUcell = myRK4GRUcell(args).to(args.device)
    top_DNN = topDNN(args.top_layers, lastbias = True).to(args.device)

    optimizer_RK4GRUcell = torch.optim.Adam(RK4GRUcell.parameters(), lr=args.lr)
    optimizer_topDNN = torch.optim.Adam(top_DNN.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(RK4GRUcell.parameters(), lr=args.lr)
    
    lr_scheduler_RK4GRUcell = torch.optim.lr_scheduler.StepLR(optimizer_RK4GRUcell,args.lr_step, args.lr_gamma)
    lr_scheduler_topDNN = torch.optim.lr_scheduler.StepLR(optimizer_topDNN,args.lr_step, args.lr_gamma)
    
    criterion = torch.nn.MSELoss(reduction = 'none')
    
    train_epochs_loss_1 = []
    train_epochs_loss_2 = []
    train_epochs_loss = []
    val_epochs_loss = []
    
    best_loss = torch.tensor(float('inf'))
    best_epoch = 0

    smallest_loss_1 = torch.tensor(float('inf'))
    smallest_loss = torch.tensor(float('inf'))
    smallest_epoch = 0
    SVj_smallest = 0

    ##training####
    for epoch in range(args.epochs):
        
        start_time = timeit.default_timer()
        RK4GRUcell.train()
        top_DNN.train()

        train_epoch_loss_1 = []
        train_epoch_loss_2 = []
        train_epoch_loss = []
        for idx, (Exc,SVi_delay, SVjtarget,T_time) in enumerate(train_dataloader):           
            Exc = Exc.to(torch.float32).to(args.device)
            SVi_delay = SVi_delay.to(torch.float32).to(args.device)
            SVjtarget = SVjtarget.to(torch.float32).to(args.device)
            SVj = torch.zeros((SVjtarget.shape[0],SVjtarget.shape[1],SVjtarget.shape[2])).to(torch.float32).to(args.device)
            T_time = T_time.to(torch.float32).to(args.device)
            
            SV_next = torch.cat( (SVi_delay[:,:,0::2].repeat(1, 1,args.layers[-1]), SVi_delay[:,:,1::2].repeat(1, 1,args.layers[-1]) ),-1)
            top_DNN_input = torch.cat((SV_next[:,:,:SV_next.shape[2]//2],T_time[:,0:1,:],Exc[:,0:1,:]),-1)
            SVj[:,0:1,:] = top_DNN(top_DNN_input)
            SVi_delay_temp = SV_next 

            for gru_s in range(gru_step - 1): 
                exci_delay = Exc[:,gru_s:(gru_s + step_delay_F + 1),:] 
                excj = Exc[:,(gru_s + step_delay_F + 1):(gru_s + step_delay_F + 2),:] 

                SV_next,_,_,_,_ = RK4GRUcell(SVi_delay_temp,step_delay_X_dX, step_delay_F,exci_delay,excj)
                top_DNN_input = torch.cat((SV_next[:,:,:SV_next.shape[2]//2],T_time[:,gru_s+1:gru_s+2,:],Exc[:,0:1,:]),-1)
                SVj[:,gru_s+1:gru_s+2,:] = top_DNN(top_DNN_input)
                SVi_delay_temp = SV_next

            #loss and optimization
            weight = torch.ones_like(SVjtarget)
            
            loss_mse = criterion(SVj, SVjtarget)
            loss_1 = torch.mean(loss_mse*weight)
            
            ####################loss constraints##################
            p_loss_2 = args.p_loss_2
            loss_2 = torch.tensor(0., device=args.device)
            for name, param in RK4GRUcell.named_parameters():
                # print(name, param.shape)
                if 'weight' in name:
                    # print(torch.sum(torch.abs(param)**p_loss_2))
                    loss_2 += torch.sum(torch.abs(param)**p_loss_2)
                if 'bias' in name:
                    # print(torch.sum(torch.abs(param)**p_loss_2))
                    loss_2 += torch.sum(torch.abs(param)**p_loss_2)
            
            for name, param in top_DNN.named_parameters():
                # print(name, param.shape)
                if 'weight' in name:
                    # print(torch.sum(torch.abs(param)**p_loss_2))
                    loss_2 += torch.sum(torch.abs(param)**p_loss_2)
                if 'bias' in name:
                    # print(torch.sum(torch.abs(param)**p_loss_2))
                    loss_2 += torch.sum(torch.abs(param)**p_loss_2)
            
            loss_2 = args.lamda_u*loss_2/(sample_size**0.5)
            ######################################################
            
            loss = loss_2 + loss_1
            
            ###############
            if (loss_1.item() < smallest_loss_1):
                smallest_loss_1 = loss_1.item()
                smallest_epoch_1 = epoch
                RK4GRUcell_model_smallest_1 = copy.deepcopy(RK4GRUcell.state_dict())
                topDNN_model_smallest_1 = copy.deepcopy(top_DNN.state_dict())
                SVj_smallest_1 = SVj.detach().numpy()
            
            if (loss.item() < smallest_loss):
                smallest_loss = loss.item()
                smallest_epoch = epoch
                RK4GRUcell_model_smallest = copy.deepcopy(RK4GRUcell.state_dict())
                topDNN_model_smallest = copy.deepcopy(top_DNN.state_dict())
                SVj_smallest = SVj.detach().numpy()
                
                ##########
            if epoch == 0 or epoch == 50 or epoch == 100 or epoch == 1000 or epoch == args.epochs-1:
                RK4GRUcell_temp = myRK4GRUcell(args).to(args.device)
                top_DNN_temp = topDNN(args.top_layers, lastbias = True).to(args.device)
                 
                RK4GRUcell_temp.load_state_dict(RK4GRUcell_model_smallest)
                top_DNN_temp.load_state_dict(topDNN_model_smallest)
                
                SVi_delay_smallest = SVi_delay.to(torch.float32).to(args.device)
                SVjtarget_smallest = SVjtarget.to(torch.float32).to(args.device)
                SVj_smallest_temp = torch.zeros((SVjtarget_smallest.shape[0],SVjtarget_smallest.shape[1],SVjtarget_smallest.shape[2])).to(torch.float32).to(args.device)
            
                SVi_delay_smallest_temp = torch.cat( (SVi_delay_smallest[:,:,0::2].repeat(1, 1,args.layers[-1]), SVi_delay_smallest[:,:,1::2].repeat(1, 1,args.layers[-1]) ),-1)
                SV_next_smallest = SVi_delay_smallest_temp
                top_DNN_temp_input = torch.cat((SV_next_smallest[:,:,:SV_next_smallest.shape[2]//2],T_time[:,0:1,:],Exc[:,0:1,:]),-1)
                SVj_smallest_temp[:,0:1,:] = top_DNN_temp(top_DNN_temp_input)
            
                for gru_s in range(gru_step - 1):
                    exci_smallest_delay = Exc[:,gru_s:(gru_s + step_delay_F + 1),:].to(torch.float32).to(args.device) 
                    excj_smallest = Exc[:,(gru_s + step_delay_F + 1):(gru_s + step_delay_F + 2),:].to(torch.float32).to(args.device) 

                    SV_next_smallest,_,_,_,_ = RK4GRUcell_temp(SVi_delay_smallest_temp,step_delay_X_dX, step_delay_F,exci_smallest_delay,excj_smallest)
                    top_DNN_temp_input = torch.cat((SV_next_smallest[:,:,:SV_next_smallest.shape[2]//2],T_time[:,gru_s+1:gru_s+2,:],Exc[:,0:1,:]),-1)
                    SVj_smallest_temp[:,gru_s+1:gru_s+2,:] = top_DNN_temp(top_DNN_temp_input)
                    SVi_delay_smallest_temp = SV_next_smallest
            
                SVj_smallest_temp = SVj_smallest_temp.detach().numpy()
                errors = np.max(np.abs(SVj_smallest_temp - SVj_smallest))
                #########
                
            ###############
            optimizer_RK4GRUcell.zero_grad()
            optimizer_topDNN.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(RK4GRUcell.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(top_DNN.parameters(), 1)
            optimizer_RK4GRUcell.step()
            optimizer_topDNN.step()
            
            train_epoch_loss.append(loss.cpu().detach().numpy())
            train_epoch_loss_1.append(loss_1.cpu().detach().numpy())
            train_epoch_loss_2.append(loss_2.cpu().detach().numpy())

        train_epochs_loss_1.append([epoch, np.average(train_epoch_loss_1)])
        train_epochs_loss_2.append([epoch, np.average(train_epoch_loss_2)])
        train_epochs_loss.append([epoch, np.average(train_epoch_loss)])
        print('###################### epoch_{} ######################'.format(epoch))
        print("[train lr_scheduler_RK4GRUcell = {}]".format( lr_scheduler_RK4GRUcell.get_last_lr()[0]))
        print("[train lr_scheduler_topDNN = {}]".format( lr_scheduler_topDNN.get_last_lr()[0]))
        print("loss_1 = {}".format(np.average(train_epoch_loss_1)))
        print("loss_2 = {}".format(np.average(train_epoch_loss_2)))
        print("loss = {}".format(np.average(train_epoch_loss)))
        
        lr_scheduler_RK4GRUcell.step()
        lr_scheduler_topDNN.step()

        # ===============================================
        #valdattion
        RK4GRUcell.eval()
        top_DNN.eval()
        val_epoch_loss = []

        for idx, (Exc,SVi_delay, SVjtarget,T_time) in enumerate(val_dataloader):
            Exc = Exc.to(torch.float32).to(args.device)
            SVi_delay = SVi_delay.to(torch.float32).to(args.device)
            SVjtarget = SVjtarget.to(torch.float32).to(args.device)
            SVj = torch.zeros((SVjtarget.shape[0],SVjtarget.shape[1],SVjtarget.shape[2])).to(torch.float32).to(args.device)
            T_time = T_time.to(torch.float32).to(args.device)

            SV_next = torch.cat( (SVi_delay[:,:,0::2].repeat(1, 1,args.layers[-1]), SVi_delay[:,:,1::2].repeat(1, 1,args.layers[-1]) ),-1)
            top_DNN_input = torch.cat((SV_next[:,:,:SV_next.shape[2]//2],T_time[:,0:1,:],Exc[:,0:1,:]),-1)
            SVj[:,0:1,:] = top_DNN(top_DNN_input)
            SVi_delay_temp = SV_next
                
            for gru_s in range(gru_step - 1): # gru_step = 200
                exci_delay = Exc[:,gru_s:(gru_s + step_delay_F + 1),:].to(torch.float32).to(args.device) 
                excj = Exc[:,(gru_s + step_delay_F + 1):(gru_s + step_delay_F + 2),:].to(torch.float32).to(args.device) 
                
                SV_next,_,_,_,_ = RK4GRUcell(SVi_delay_temp,step_delay_X_dX, step_delay_F,exci_delay,excj)
                top_DNN_input = torch.cat((SV_next[:,:,:SV_next.shape[2]//2],T_time[:,gru_s+1:gru_s+2,:],Exc[:,0:1,:]),-1)
                SVj[:,gru_s+1:gru_s+2,:] = top_DNN(top_DNN_input)
                SVi_delay_temp = SV_next
            
            # loss              
            weight = torch.ones_like(SVjtarget)

            loss_mse = criterion(SVj, SVjtarget)
            loss = torch.mean(loss_mse*weight)
            val_epoch_loss.append(loss.cpu().detach().numpy() )

        if (np.average(val_epoch_loss) < best_loss):
            best_loss = np.average(val_epoch_loss)
            RK4GRUcell_model_best = copy.deepcopy(RK4GRUcell.state_dict())
            topDNN_model_best = copy.deepcopy(top_DNN.state_dict())
            best_epoch = epoch
        
        val_epochs_loss.append([epoch, np.average(val_epoch_loss)])            
        print("[val]  loss = {}".format(np.average(val_epoch_loss)), end='\n')
        
        end_time = timeit.default_timer()
        time_consume = end_time - start_time
        print(f'best_epoch is {best_epoch}')
        print(f'best_loss is {best_loss:.6f}')
        print(f'smallest_epoch_1 is {smallest_epoch_1}')
        print(f'smallest_loss_1 is {smallest_loss_1:.6f}')
        print(f'smallest_epoch is {smallest_epoch}')
        print(f'smallest_loss is {smallest_loss:.6f}')
        print(f'errors is {errors:.6f}')
        print(f'Consumed time is {time_consume:.3f} s')
        print(' ')
    ##########################################################################################################
    RK4GRUcell_model_last = RK4GRUcell.state_dict()
    topDNN_model_last = top_DNN.state_dict()

    str_layers = '_'.join(map(str, args.layers))
    str_top_layers = '_'.join(map(str, args.top_layers))

    path_save_model_best = modelsave_path + 'RK4GRUcell_best_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.pth'
    torch.save(RK4GRUcell_model_best, path_save_model_best)

    path_save_model_last = modelsave_path + 'RK4GRUcell_last_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.pth'
    torch.save(RK4GRUcell_model_last, path_save_model_last)
    
    path_save_model_smallest_1 = modelsave_path + 'RK4GRUcell_smallest_1_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.pth'
    torch.save(RK4GRUcell_model_smallest_1, path_save_model_smallest_1)
    
    path_save_model_smallest = modelsave_path + 'RK4GRUcell_smallest_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.pth'
    torch.save(RK4GRUcell_model_smallest, path_save_model_smallest)

    path_save_model_best = modelsave_path + 'topDNN_best_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.pth'
    torch.save(topDNN_model_best, path_save_model_best)

    path_save_model_last = modelsave_path + 'topDNN_last_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.pth'
    torch.save(topDNN_model_last, path_save_model_last)

    path_save_model_smallest_1 = modelsave_path + 'topDNN_smallest_1_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.pth'
    torch.save(topDNN_model_smallest_1, path_save_model_smallest_1)

    path_save_model_smallest = modelsave_path + 'topDNN_smallest_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.pth'
    torch.save(topDNN_model_smallest, path_save_model_smallest)
   
    train_epochs_loss_1 = np.array(train_epochs_loss_1)
    train_epochs_loss_2 = np.array(train_epochs_loss_2)
    train_epochs_loss = np.array(train_epochs_loss)
    val_epochs_loss = np.array(val_epochs_loss)
    
    path_save_train_epochs_loss_1 = modelsave_path + 'train_epochs_loss_1_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.npy'
    path_save_train_epochs_loss_2 = modelsave_path + 'train_epochs_loss_2_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.npy'
    path_save_train_epochs_loss = modelsave_path + 'train_epochs_loss_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.npy'
    path_save_val_epochs_loss = modelsave_path + 'val_epochs_loss_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.npy'
    np.save(path_save_train_epochs_loss_1, train_epochs_loss_1)
    np.save(path_save_train_epochs_loss_2, train_epochs_loss_2)
    np.save(path_save_train_epochs_loss, train_epochs_loss)
    np.save(path_save_val_epochs_loss, val_epochs_loss)
    
    Train_epochs_loss_1 = {'Train_epochs_loss_1':train_epochs_loss_1}
    Train_epochs_loss_2 = {'Train_epochs_loss_2':train_epochs_loss_2}
    Train_epochs_loss = {'Train_epochs_loss':train_epochs_loss}
    Val_epochs_loss = {'Val_epochs_loss':val_epochs_loss}
    scipy.io.savemat(data_path + 'Train_epochs_loss_1.mat', Train_epochs_loss_1)
    scipy.io.savemat(data_path + 'Train_epochs_loss_2.mat', Train_epochs_loss_2)
    scipy.io.savemat(data_path + 'Train_epochs_loss.mat', Train_epochs_loss)
    scipy.io.savemat(data_path + 'Val_epochs_loss.mat', Val_epochs_loss)

    path_save_train_indices = modelsave_path + 'train_indices_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.npy'
    path_save_val_indices = modelsave_path + 'val_indices_' + str(gru_step) + '_' + str(num_sample['Sample_size'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.npy'
    np.save(path_save_train_indices, train_indices)
    np.save(path_save_val_indices, val_indices)

    X_smallest = np.transpose(SVj_smallest[:,:,0], axes = [1,0])
    X_smallest = {'X_smallest':X_smallest  } 
    scipy.io.savemat(data_path + 'X_smallest.mat', X_smallest) 

    return RK4GRUcell_model_last,topDNN_model_last,train_epochs_loss,val_epochs_loss

###############################################################################
###############################################################################
#Training Module

args = Args()
train_RK4PIGRU_main(args)

