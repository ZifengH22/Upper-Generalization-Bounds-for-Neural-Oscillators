clear;clc;close all;

%%%%%%%%%%%loading data%%%%%%%%%%%
current_path = cd;
dir_info = dir(current_path);
dir_info = dir_info([dir_info.isdir]); % Keep only directories
dir_info = dir_info(~ismember({dir_info.name}, {'.', '..'})); % Exclude 
dir_names = {dir_info.name};

file_names_suffix = {'Case_2_ReLU_40_100_';'Case_2_ReLU_40_200_';'Case_2_ReLU_40_400_';'Case_2_ReLU_40_800_';'Case_2_ReLU_40_1600_';'Case_2_ReLU_40_3200_'};
[number_filefolder,~] = size(file_names_suffix);

for i = 1:number_filefolder
    file_name_suffix_l = cellstr(file_names_suffix{i});
    matchingIdx = find(contains(dir_names, file_name_suffix_l) == 1);

    %%%orignal
    matchingIdx_original = matchingIdx(1);
    file_name_l = [current_path,'\',dir_names{matchingIdx_original}];
    file_name_l_data = [file_name_l,'\data'];
    file_name_l_data_X_train = [file_name_l_data,'\X_train.mat'];
    file_name_l_data_X_pred = [file_name_l_data,'\X_pred.mat'];
    file_name_l_data_Train_epochs_loss = [file_name_l_data,'\Train_epochs_loss.mat'];
    file_name_l_data_Val_epochs_loss = [file_name_l_data,'\Val_epochs_loss.mat'];

    assignin('base', ['X_train_',file_names_suffix{i}([6:end])], load(file_name_l_data_X_train).X_train);
    assignin('base', ['X_pred_',file_names_suffix{i}([6:end])], load(file_name_l_data_X_pred).X_pred);
    assignin('base', ['Train_epochs_loss_',file_names_suffix{i}([6:end])], load(file_name_l_data_Train_epochs_loss).Train_epochs_loss);
    assignin('base', ['Val_epochs_loss_',file_names_suffix{i}([6:end])], load(file_name_l_data_Val_epochs_loss).Val_epochs_loss);

    %%%L1
    matchingIdx_L1 = matchingIdx(2);
    file_name_l = [current_path,'\',dir_names{matchingIdx_L1}];
    file_name_l_data = [file_name_l,'\data'];
    file_name_l_data_X_train = [file_name_l_data,'\X_train.mat'];
    file_name_l_data_X_pred = [file_name_l_data,'\X_pred.mat'];
    file_name_l_data_Train_epochs_loss = [file_name_l_data,'\Train_epochs_loss.mat'];
    file_name_l_data_Val_epochs_loss = [file_name_l_data,'\Val_epochs_loss.mat'];

    assignin('base', ['X_pred_L1_',file_names_suffix{i}([6:end])], load(file_name_l_data_X_pred).X_pred);
    assignin('base', ['Train_epochs_loss_L1_',file_names_suffix{i}([6:end])], load(file_name_l_data_Train_epochs_loss).Train_epochs_loss);
    assignin('base', ['Val_epochs_loss_L1_',file_names_suffix{i}([6:end])], load(file_name_l_data_Val_epochs_loss).Val_epochs_loss);
end

%%%%%%%%%%%calculation and plot%%%%%%%%%%%
sample = [100,200,400,800,1600,3200];
error_mse_data = zeros(size(sample));
error_max_data = zeros(size(sample));
for i = 1:number_filefolder
    % commandstr_error_mse_data = ['error_mse_data(i) = ','mean(mean((',['X_train_',file_names_suffix{i}([6:end])], '-', ['X_pred_',file_names_suffix{i}([6:end])],').^2));'];
    commandstr_error_mse_data = ['error_mse_data(i) = ','mean(mean((',['X_train_',file_names_suffix{i}([6:end])], '-', ['X_pred_',file_names_suffix{i}([6:end])],').^2))/','mean(mean((',['X_train_',file_names_suffix{i}([6:end])],').^2))',';'];
    eval(commandstr_error_mse_data);

    % commandstr_error_max_data = ['error_max_data(i) = ','max(max(abs(',['X_train_',file_names_suffix{i}([6:end])], '-', ['X_pred_',file_names_suffix{i}([6:end])],')));'];
    commandstr_error_max_data = ['error_max_data(i) = ','max(max(abs(',['X_train_',file_names_suffix{i}([6:end])], '-', ['X_pred_',file_names_suffix{i}([6:end])],')))/','max(max(abs(',['X_train_',file_names_suffix{i}([6:end])],')));'];
    eval(commandstr_error_max_data);
end

error_mse_data_L1 = zeros(size(sample));
error_max_data_L1 = zeros(size(sample));
for i = 1:number_filefolder
    % commandstr_error_mse_data = ['error_mse_data_L1(i) = ','mean(mean((',['X_train_',file_names_suffix{i}([6:end])], '-', ['X_pred_L1_',file_names_suffix{i}([6:end])],').^2));'];
    commandstr_error_mse_data = ['error_mse_data_L1(i) = ','mean(mean((',['X_train_',file_names_suffix{i}([6:end])], '-', ['X_pred_L1_',file_names_suffix{i}([6:end])],').^2))/','mean(mean((',['X_train_',file_names_suffix{i}([6:end])],').^2))',';'];
    eval(commandstr_error_mse_data);

    % commandstr_error_max_data = ['error_max_data_L1(i) = ','max(max(abs(',['X_train_',file_names_suffix{i}([6:end])], '-', ['X_pred_L1_',file_names_suffix{i}([6:end])],')));'];
    commandstr_error_max_data = ['error_max_data_L1(i) = ','max(max(abs(',['X_train_',file_names_suffix{i}([6:end])], '-', ['X_pred_L1_',file_names_suffix{i}([6:end])],')))/','max(max(abs(',['X_train_',file_names_suffix{i}([6:end])],')));'];
    eval(commandstr_error_max_data);
end

a = 0;
b = 4;
x_sample = linspace(sample(1),sample(end),50);
error_max_ana = a+b*x_sample.^(-0.5);
figure(21)
loglog(sample,error_max_data,'r*',sample,error_max_data_L1,'bo',x_sample,error_max_ana,'k-');
xlabel('$N$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{X,\infty}$', 'Interpreter', 'latex');
legend('Numerical results from original loss','Numerical results from L1-constrained loss', ['$\tilde{\varepsilon}_{X,\infty} = ',num2str(b),'N^{-0.5}_{\mathrm{Sample}}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);
set(gca, 'XTick', [100, 200, 400, 800, 1600, 3200]);
% exportgraphics(gcf,'Fig_3_Relative_supremum_error_of_U_e_to_X_5.pdf.pdf','Resolution',300);
% savefig('Fig_3_Relative_supremum_error_of_U_e_to_X_5.pdf.fig');

a_1 = 0;
b_1 = 3500;
c_1 = -2.2;
a_2 = a_1;
b_2 = 0.13;
c_2 = -0.5;
x_sample = linspace(sample(1),sample(end),50);
error_mse_ana_1 = a_1+b_1*x_sample.^(c_1);
error_mse_ana_2 = a_2+b_2*x_sample.^(c_2);
error_mse_ana = max(error_mse_ana_1,error_mse_ana_2);
[~,ind_smallest] = min(abs(error_mse_ana_1 - error_mse_ana_2));
error_mse_ana_1 = error_mse_ana_1(1:ind_smallest);
x_sample_1 = x_sample(1:ind_smallest);
error_mse_ana_2 = error_mse_ana_2(ind_smallest:end);
x_sample_2 = x_sample(ind_smallest:end);

figure(22)
loglog(sample,error_mse_data,'r*',sample,error_mse_data_L1,'bo',x_sample_1,error_mse_ana_1,'r--',x_sample_2,error_mse_ana_2,'b-');
xlabel('$N$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{X,2}$', 'Interpreter', 'latex');
legend('Numerical results from $\hat\ell_{\lambda_L}$ with $\lambda_L = 0$','Numerical results from $\hat\ell_{\lambda_L}$ with $\lambda_L = 0.003$', ['$\tilde{\varepsilon}_{X,2} = ',num2str(b_1),'N^{',num2str(c_1),'}$'], ['$\tilde{\varepsilon}_{X,2} = ',num2str(b_2),'N^{',num2str(c_2),'}$'], 'Interpreter', 'latex');
ylim([0,1])
xlim([0,3200])
set(gca, 'XTick', [100, 200, 400, 800, 1600, 3200]);
set(gca,'fontsize',15);
set(gcf, 'Position', [100 100 550 400]);           
set(gcf, 'PaperPosition', [0 0 5.5 4]);            
set(gcf, 'PaperSize', [5.5 4]);
% exportgraphics(gcf,'Fig_1_Relative_error_of_U_e_to_X_5.pdf','Resolution',300);
savefig('Fig_1_Relative_error_of_U_e_to_X_5.fig');

%%%%%%%%%%%convergence error%%%%%%%%%%%%%%%%%%
figure(23)
plot(Train_epochs_loss_2_ReLU_40_100_(:,1),Train_epochs_loss_2_ReLU_40_100_(:,2),'b-',Val_epochs_loss_2_ReLU_40_100_(:,1),Val_epochs_loss_2_ReLU_40_100_(:,2),'r-.');
hold on;
plot(Train_epochs_loss_2_ReLU_40_200_(:,1),Train_epochs_loss_2_ReLU_40_200_(:,2),'b-',Val_epochs_loss_2_ReLU_40_200_(:,1),Val_epochs_loss_2_ReLU_40_200_(:,2),'r-.');
plot(Train_epochs_loss_2_ReLU_40_400_(:,1),Train_epochs_loss_2_ReLU_40_400_(:,2),'b-',Val_epochs_loss_2_ReLU_40_400_(:,1),Val_epochs_loss_2_ReLU_40_400_(:,2),'r-.');
plot(Train_epochs_loss_2_ReLU_40_800_(:,1),Train_epochs_loss_2_ReLU_40_800_(:,2),'b-',Val_epochs_loss_2_ReLU_40_800_(:,1),Val_epochs_loss_2_ReLU_40_800_(:,2),'r-.');
plot(Train_epochs_loss_2_ReLU_40_1600_(:,1),Train_epochs_loss_2_ReLU_40_1600_(:,2),'b-',Val_epochs_loss_2_ReLU_40_1600_(:,1),Val_epochs_loss_2_ReLU_40_1600_(:,2),'r-.');
plot(Train_epochs_loss_2_ReLU_40_3200_(:,1),Train_epochs_loss_2_ReLU_40_3200_(:,2),'b-',Val_epochs_loss_2_ReLU_40_3200_(:,1),Val_epochs_loss_2_ReLU_40_3200_(:,2),'r-.');
legend('Training loss','Validation loss', 'Interpreter', 'latex');
xlim([0,1000]);
ylim([1e-2,1e3]);
xlabel('Epoch', 'Interpreter', 'latex');
ylabel(['$\ell_{', num2str(2), '}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);
set(gca,'XScale','linear','YScale','log');