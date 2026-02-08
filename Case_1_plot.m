clear;clc;close all;

%%%%%%%%%%%loading data%%%%%%%%%%%
current_path = cd;
Data_path = [current_path,'\Case_1_time_length\data\'];

dir_info = dir([current_path,'\Case_1_time_length\']);
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
    file_name_l = [current_path,'\Case_1_time_length\',dir_names{matchingIdx_original}];
    file_name_l_data = [file_name_l,'\data'];
    % file_name_l_data_X_train = [file_name_l_data,'\E_X_train.mat'];
    % file_name_l_data_X_pred = [file_name_l_data,'\E_X_pred.mat'];
    file_name_l_data_Train_epochs_loss = [file_name_l_data,'\Train_epochs_loss.mat'];
    file_name_l_data_Val_epochs_loss = [file_name_l_data,'\Val_epochs_loss.mat'];

    % assignin('base', ['E_X_train_',file_names_suffix{i}([6:end])], load(file_name_l_data_X_train).E_X_train);
    % assignin('base', ['E_X_pred_',file_names_suffix{i}([6:end])], load(file_name_l_data_X_pred).E_X_pred);
    assignin('base', ['Train_epochs_loss_',file_names_suffix{i}([6:end])], load(file_name_l_data_Train_epochs_loss).Train_epochs_loss);
    assignin('base', ['Val_epochs_loss_',file_names_suffix{i}([6:end])], load(file_name_l_data_Val_epochs_loss).Val_epochs_loss);
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
b = 1.8e-5;
c = 1.5;
time_length = linspace(num_time_element,num_time_element*6,50)*dt;
time_length_sample = [1:6]*num_time_element*dt;
error_ana = a+b*time_length.^(c);
figure(11)
plot(time_length_sample,error_mse,'r*',time_length,error_ana,'k-');
xlabel('$T(s)$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{E,2}$', 'Interpreter', 'latex');
legend('Numerical results',['$\tilde{\varepsilon}_{E,2} = ','1.8\times10^{-5}','T^{',num2str(c),'}$'], 'Interpreter', 'latex','Location','northwest');
set(gca,'fontsize',15);
set(gcf, 'Position', [100 100 550 400]);           
set(gcf, 'PaperPosition', [0 0 5.5 4]);            
set(gcf, 'PaperSize', [5.5 4]);
% exportgraphics(gcf,'Fig_2_Relative_error_of_U_e_to_EX_5.pdf','Resolution',300);
savefig('Fig_2_Relative_error_of_U_e_to_EX_5.fig');

figure(12)
plot(Train_epochs_loss_path_time_length_1000(:,1),Train_epochs_loss_path_time_length_1000(:,2),'b-',Val_epochs_loss_path_time_length_1000(:,1),Val_epochs_loss_path_time_length_1000(:,2),'r-.');
hold on;
plot(Train_epochs_loss_path_time_length_1500(:,1),Train_epochs_loss_path_time_length_1500(:,2),'b-',Val_epochs_loss_path_time_length_1500(:,1),Val_epochs_loss_path_time_length_1500(:,2),'r-.');
plot(Train_epochs_loss_path_time_length_2000(:,1),Train_epochs_loss_path_time_length_2000(:,2),'b-',Val_epochs_loss_path_time_length_2000(:,1),Val_epochs_loss_path_time_length_2000(:,2),'r-.');
plot(Train_epochs_loss_path_time_length_2500(:,1),Train_epochs_loss_path_time_length_2500(:,2),'b-',Val_epochs_loss_path_time_length_2500(:,1),Val_epochs_loss_path_time_length_2500(:,2),'r-.');
plot(Train_epochs_loss_path_time_length_3000(:,1),Train_epochs_loss_path_time_length_3000(:,2),'b-',Val_epochs_loss_path_time_length_3000(:,1),Val_epochs_loss_path_time_length_3000(:,2),'r-.');
legend('Training loss','Validation loss', 'Interpreter', 'latex');
xlim([0,3000]);
ylim([1e-2,1e3]);
xlabel('Epoch', 'Interpreter', 'latex');
ylabel(['$\ell_{', num2str(2), '}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);
set(gcf, 'Position', [100 100 550 400]);           
set(gcf, 'PaperPosition', [0 0 5.5 4]);            
set(gcf, 'PaperSize', [5.5 4]);
set(gca,'XScale','linear','YScale','log');

%%%%%%%%%PDF and CDF of EX%%%%%%%%%%%%
E_X_range = linspace(1,33,100);
[PDF_E_X_pred,~] = ksdensity(E_X_pred_3000(end,:),E_X_range,'Bandwidth',0.1);
[PDF_E_X1,~] = ksdensity(E_X1(end,:),E_X_range,'Bandwidth',0.1);

figure(3)
plot(E_X_range,PDF_E_X_pred,'k-',E_X_range,PDF_E_X1,'b--');
xlim([0,35]);
legend('$\it{E_{X_\mathrm{5},l}(\mathrm{30})}$','$\it{\tilde{E}_{X_\mathrm{5},l}(\mathrm{30})}$', 'Interpreter', 'latex');
xlabel('$\it{E_{X_\mathrm{5}}(\mathrm{30})}$', 'Interpreter', 'latex');
ylabel('PDF', 'Interpreter', 'latex');
set(gca,'fontsize',15);
set(gcf, 'Position', [100 100 550 400]);           
set(gcf, 'PaperPosition', [0 0 5.5 4]);            
set(gcf, 'PaperSize', [5.5 4]);
set(gca,'yscale','linear');
savefig('Fig_3_PDF_EX_30.fig');

[CDF_E_X_pred,E_X_pred_range] = ecdf(E_X_pred_3000(end,:));
[CDF_E_X1,E_X1_range] = ecdf(E_X1(end,:));
CDF_E_X1_y = CDF_E_X1;

eps = 1e-5;
CDF_E_X_pred = min(max(CDF_E_X_pred, eps), 1 - eps);
CDF_E_X1 = min(max(CDF_E_X1, eps), 1 - eps);
CDF_E_X1_y = min(max(CDF_E_X1_y, eps/2), 1 - eps/2);

z_pred = norminv(CDF_E_X_pred);
z_1 = norminv(CDF_E_X1);
z_1_y = norminv(CDF_E_X1_y);
z_1_y_5 = linspace(z_1_y(1),z_1_y(end),5);
idx = zeros(size(z_1_y_5)); 
for i = 1:length(z_1_y_5)
    [~, idx(i)] = min(abs(z_1_y - z_1_y_5(i)));
end
prob_labels = CDF_E_X1_y(idx);

figure(4)
plot(E_X_pred_range,z_pred,'bo',E_X1_range,z_1,'r*');
xlim([0,35]);
legend('$\it{E_{X_\mathrm{5},l}(\mathrm{30})}$','$\it{\tilde{E}_{X_\mathrm{5},l}(\mathrm{30})}$', 'Interpreter', 'latex','Location','northwest');
xlabel('$\it{E_{X_\mathrm{5}}(\mathrm{30})}$', 'Interpreter', 'latex');
ylabel('CDF', 'Interpreter', 'latex');
yticks(z_1_y_5);
yticklabels(compose('%.3f', prob_labels));
ylim([min(z_1_y), max(z_1_y)]);
set(gca,'fontsize',15);
set(gcf, 'Position', [100 100 550 400]);           
set(gcf, 'PaperPosition', [0 0 5.5 4]);            
set(gcf, 'PaperSize', [5.5 4]);
set(gca,'xscale','linear','yscale','linear');
savefig('Fig_4_CDF_EX_30.fig');

