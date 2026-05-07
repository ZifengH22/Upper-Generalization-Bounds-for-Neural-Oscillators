clear;clc;
% close all;
current_path = cd;
Data_path = [current_path,'\data\'];
Data_path_sample_100 = [current_path,'\Data_path_sample_100\'];
Data_path_sample_200 = [current_path,'\Data_path_sample_200\'];
Data_path_sample_400 = [current_path,'\Data_path_sample_400\'];
Data_path_sample_800 = [current_path,'\Data_path_sample_800\'];
Data_path_sample_1600 = [current_path,'\Data_path_sample_1600\'];
Data_path_sample_3200 = [current_path,'\Data_path_sample_3200\'];

seed = 1028; %[1028,1128,1228,1328];
Sample_size_vector = [100,200,400,800,1600,3200];
Sample_size_number = length(Sample_size_vector);
rng(seed);

%%%%%%%%%%
load M_response.mat;
load K_response.mat;
load C_response.mat;
load K_in_response.mat;
beta = 2;
gamma_r = 2;
s = 3;
lamda = 0.01;


%%%%%%checking data%%%%%%%%%
index_to_check = 1;
temp_position = ['Data_path_sample_',num2str(round(Sample_size_vector(index_to_check))),'\data\'];
load([temp_position,'t_train.mat']);
load([temp_position,'F_train_',num2str(seed),'.mat']);
load([temp_position,'X_dX_input_train.mat']);
load([temp_position,'X_dX_output_train_',num2str(seed),'.mat']);
% load([temp_position,'E_X_output_train_',num2str(seed),'.mat']);

t = t_train(1,:);
dt = t(2)-t(1);
[X_check,dX_check,ddX_check,Z_check,~] = Y_response(t,dt,M,C,K_in,beta,gamma_r,s,lamda,F_train');
% E_X_check = Ex_extreme_value_process(X_check);

index = 100*2^(index_to_check-1)
figure(1)
plot(t,X_dX_output_train(index,:),'b.',t,X_check(:,index),'k--');
xlabel('Time (s)');
ylabel('Response X');
legend('Training data','Checking data');
% figure(2)
% plot(t,abs(X_dX_output_train(index,:)),'b.-',t,E_X_check(:,index),'k--');
% xlabel('Time (s)');
% ylabel('Response E_X');
% legend('Training data','Checking data');
