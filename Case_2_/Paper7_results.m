clear;clc;close all;

%%%%%%%%%%%loading data%%%%%%%%%%%
current_path = cd;
Data_path = [current_path,'\data\'];

dir_info = dir(current_path);
dir_info = dir_info([dir_info.isdir]); % Keep only directories
dir_info = dir_info(~ismember({dir_info.name}, {'.', '..'})); % Exclude 
dir_names = {dir_info.name};

% file_names_suffix = {'Data_path_time_length_1000';'Data_path_time_length_2000';'Data_path_time_length_3000';'Data_path_time_length_4000'};
file_names_suffix = {'Data_path_time_length_500';'Data_path_time_length_1000';'Data_path_time_length_1500';'Data_path_time_length_2000';'Data_path_time_length_2500';'Data_path_time_length_3000'};
[number_filefolder,~] = size(file_names_suffix);
for i = 1:number_filefolder
    file_name_suffix_l = cellstr(file_names_suffix{i});
    matchingIdx = find(contains(dir_names, file_name_suffix_l) == 1);

    %%%orignal
    matchingIdx_original = matchingIdx(1);
    file_name_l = [current_path,'\',dir_names{matchingIdx_original}];
    file_name_l_data = [file_name_l,'\data'];
    % file_name_l_data_X_train = [file_name_l_data,'\E_X_train.mat'];
    % file_name_l_data_X_pred = [file_name_l_data,'\E_X_pred.mat'];
    file_name_l_data_Train_epochs_loss = [file_name_l_data,'\Train_epochs_loss.mat'];
    file_name_l_data_Val_epochs_loss = [file_name_l_data,'\Val_epochs_loss.mat'];

    % assignin('base', ['E_X_train_',file_names_suffix{i}([6:end])], load(file_name_l_data_X_train).E_X_train);
    % assignin('base', ['E_X_pred_',file_names_suffix{i}([6:end])], load(file_name_l_data_X_pred).E_X_pred);
    % assignin('base', ['Train_epochs_loss_',file_names_suffix{i}([6:end])], load(file_name_l_data_Train_epochs_loss).Train_epochs_loss);
    % assignin('base', ['Val_epochs_loss_',file_names_suffix{i}([6:end])], load(file_name_l_data_Val_epochs_loss).Val_epochs_loss);
end

num_time_element = 500;
for i = 1:6
    file_name_l_data_X_pred = [Data_path,'E_X_pred_',num2str(round(i*num_time_element)),'.mat'];
    load(file_name_l_data_X_pred);
end
file_name_l_data_X_train = [Data_path,'E_X1_response','.mat'];
load(file_name_l_data_X_train);


%%%%%%%%%%%calculation and plot%%%%%%%%%%%
[num_time,num_sample] = size(E_X1);
dt = 0.01;
t = 0:dt:(num_time-1)*dt;
E_X1 = E_X1(1:6*num_time_element,:);

error_mse = zeros(1,6);
for i = 1:6
    commandstr_error_mse_data = ['error_mse(i) = ','mean(sum((','E_X1(1:i*num_time_element,:)', '-', ['E_X_pred_',num2str(round(i*num_time_element))],').^2))/','mean(sum(E_X1.^2))',';'];
    eval(commandstr_error_mse_data);
end

a = 0;
b = 3.2e-5;
c = 1.5;
time_length = linspace(num_time_element,num_time_element*6,50)*dt;
time_length_sample = [1:6]*num_time_element*dt;
error_ana = a+b*time_length.^(c);
figure(1)
plot(time_length_sample,error_mse,'r*',time_length,error_ana,'k-');
xlabel('$T(s)$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{E_X,2}$', 'Interpreter', 'latex');
legend('Numerical results',['$\tilde{\varepsilon}_{E,2} = ','3.2\times10^{-5}','T^{',num2str(c),'}$'], 'Interpreter', 'latex','Location','northwest');
set(gca,'fontsize',15);

% figure(2)
% plot(Train_epochs_loss_path_time_length_1000(:,1),Train_epochs_loss_path_time_length_1000(:,2),'b-',Val_epochs_loss_path_time_length_1000(:,1),Val_epochs_loss_path_time_length_1000(:,2),'r-.');
% hold on;
% plot(Train_epochs_loss_path_time_length_1500(:,1),Train_epochs_loss_path_time_length_1500(:,2),'b-',Val_epochs_loss_path_time_length_1500(:,1),Val_epochs_loss_path_time_length_1500(:,2),'r-.');
% plot(Train_epochs_loss_path_time_length_2000(:,1),Train_epochs_loss_path_time_length_2000(:,2),'b-',Val_epochs_loss_path_time_length_2000(:,1),Val_epochs_loss_path_time_length_2000(:,2),'r-.');
% plot(Train_epochs_loss_path_time_length_2500(:,1),Train_epochs_loss_path_time_length_2500(:,2),'b-',Val_epochs_loss_path_time_length_2500(:,1),Val_epochs_loss_path_time_length_2500(:,2),'r-.');
% plot(Train_epochs_loss_path_time_length_3000(:,1),Train_epochs_loss_path_time_length_3000(:,2),'b-',Val_epochs_loss_path_time_length_3000(:,1),Val_epochs_loss_path_time_length_3000(:,2),'r-.');
% legend('Training loss','Validation loss', 'Interpreter', 'latex');
% xlim([0,3000]);
% ylim([1e-2,1e3]);
% xlabel('Epoch', 'Interpreter', 'latex');
% ylabel(['$\ell_{', num2str(2), '}$'], 'Interpreter', 'latex');
% set(gca,'fontsize',15);
% set(gca,'XScale','linear','YScale','log');