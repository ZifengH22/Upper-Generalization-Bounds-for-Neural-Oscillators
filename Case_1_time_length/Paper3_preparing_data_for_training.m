clear;clc;close all;
current_path = cd;
Data_path = [current_path,'\data\'];
Data_path_time_length_500 = [current_path,'\time_length_500\'];
Data_path_time_length_1000 = [current_path,'\time_length_1000\'];
Data_path_time_length_1500 = [current_path,'\time_length_1500\'];
Data_path_time_length_2000 = [current_path,'\time_length_2000\'];
Data_path_time_length_2500 = [current_path,'\time_length_2500\'];
Data_path_time_length_3000 = [current_path,'\time_length_3000\'];

rng(1228);
Sample_size = 1600;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load([Data_path,'Acc.mat']);
load([Data_path,'X1_response.mat']);
load([Data_path,'E_X1_response.mat']);

X_l = X1;
E_X_l = E_X1;

[num_time,num_sample] = size(X_l);
dt = 0.01;
t = 0:dt:(num_time-1)*dt;
num_time_element = 500;
num_case = round(3000/num_time_element);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Acc_train = Acc;
X_train = X_l;
E_X_train = E_X_l;

num_time_select = num_time;  % 训练序列长度
num_sample_select = Sample_size*1.25; % 训练整序列个数

F_train_all = zeros(num_sample_select,num_time_select,1);
X_dX_input_train_all = zeros(num_sample_select,1,2);
X_dX_output_train_all = zeros(num_sample_select,num_time_select,1);
E_X_output_train_all = zeros(num_sample_select,num_time_select,1);

for i = 1:num_sample_select
    F_l = Acc_train(:, i);
    F_train_all(i,:,:) = F_l;
    X_dX_l = X_train(:,i);
    X_dX_output_train_all(i,:,:) = X_dX_l;
    E_X_l = E_X_train(:,i);
    E_X_output_train_all(i,:,:) = E_X_l;
end

t_train_all = repmat(t,num_sample_select,1,1);

X_output_train_all = squeeze(X_dX_output_train_all(:,:,1));
dX_output_train_all_app = diff(X_output_train_all')'/dt;
ddX_output_train_all_app = diff(dX_output_train_all_app')'/dt;

coef_F = std(F_train_all(:));
coef_X_output = std(X_output_train_all(:));
coef_dX_output = std(dX_output_train_all_app(:));
coef_ddX_output = std(ddX_output_train_all_app(:));
coef_E_X_output = std(E_X_output_train_all(:));

numTrain = size(X_dX_input_train_all,1); %总训练样本数

%%%%%%%%%%%%%%%%%%%%%%%%%%%%saving data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:num_case
    temp_position = ['Data_path_time_length_',num2str(round(i*num_time_element)),'\data\'];
    mkdir(temp_position)
    % cd(temp_position)
    disp(num2str(i))
    
    F_train = F_train_all(:,1:num_time_element*i,1);
    X_dX_output_train = X_dX_output_train_all(:,1:num_time_element*i,1);
    X_dX_input_train = X_dX_input_train_all;
    E_X_output_train = E_X_output_train_all(:,1:num_time_element*i,1);
    t_train = t_train_all(:,1:num_time_element*i,1);

    save([temp_position,'t_train.mat'],'t_train', '-v7.3');
    save([temp_position,'F_train.mat'],'F_train', '-v7.3');
    save([temp_position,'X_dX_input_train.mat'],'X_dX_input_train', '-v7.3');
    save([temp_position,'X_dX_output_train.mat'],'X_dX_output_train', '-v7.3');
    save([temp_position,'E_X_output_train.mat'],'E_X_output_train', '-v7.3');

    save([temp_position,'num_sample_select.mat'],'num_sample_select');
    save([temp_position,'Sample_size.mat'],'Sample_size');
    save([temp_position,'coef_F.mat'],'coef_F');
    save([temp_position,'coef_X_output.mat'],'coef_X_output');
    save([temp_position,'coef_dX_output.mat'],'coef_dX_output');
    save([temp_position,'coef_ddX_output.mat'],'coef_ddX_output');
    save([temp_position,'coef_E_X_output.mat'],'coef_E_X_output');   
end

disp([' '])
disp(['The number of trained whole time series is: ',num2str(num_sample_select)])
disp([' '])
disp(['The total length of each trained sample is: ',num2str(num_time_select)])
disp([' '])
disp(['The total number of trained samples is: ',num2str(numTrain)])
