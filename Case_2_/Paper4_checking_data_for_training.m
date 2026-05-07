clear;clc;
% close all;
current_path = cd;
Data_path = [current_path,'\data\'];
Data_path_time_length_500 = [current_path,'\time_length_500\'];
Data_path_time_length_1000 = [current_path,'\time_length_1000\'];
Data_path_time_length_1500 = [current_path,'\time_length_1500\'];
Data_path_time_length_2000 = [current_path,'\time_length_2000\'];
Data_path_time_length_2500 = [current_path,'\time_length_2500\'];
Data_path_time_length_3000 = [current_path,'\time_length_3000\'];

seed = 1528; %[1228,1328,1428,1528];

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
num_time_element = 500;
index_to_check = 6;
temp_position = ['Data_path_time_length_',num2str(round(index_to_check*num_time_element)),'\data\'];
load([temp_position,'t_train.mat']);
load([temp_position,'F_train_',num2str(seed),'.mat']);
load([temp_position,'X_dX_input_train.mat']);
load([temp_position,'X_dX_output_train_',num2str(seed),'.mat']);
load([temp_position,'E_X_output_train_',num2str(seed),'.mat']);

t = t_train(1,:);
dt = t(2)-t(1);
[X_check,dX_check,ddX_check,Z_check,~] = Y_response(t,dt,M,C,K_in,beta,gamma_r,s,lamda,F_train');
E_X_check = Ex_extreme_value_process(X_check);

index = 1445;
figure(1)
plot(t,X_dX_output_train(index,:),'b.-',t,X_check(:,index),'k--');
xlabel('Time (s)');
ylabel('Response X');
legend('Training data','Checking data');
figure(2)
plot(t,abs(X_dX_output_train(index,:)),'b.-',t,E_X_check(:,index),'k--');
xlabel('Time (s)');
ylabel('Response E_X');
legend('Training data','Checking data');
