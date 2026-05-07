clear;clc;close all;

%%%%%%%%%%%loading data%%%%%%%%%%%
current_path = cd;
Data_path = [current_path,'\Case_2_\data\'];

length_vector = [500,1000,1500,2000,2500,3000];
length_vector_number = length(length_vector);

%%%%%%%%%%%calculation and plot%%%%%%%%%%
num_time_element = 500;
file_name_l_data_X_train = [Data_path,'E_X1_response','.mat'];
load(file_name_l_data_X_train);
[num_time,num_sample] = size(E_X1);
dt = 0.01;
t = 0:dt:(num_time-1)*dt;
E_X1 = E_X1(1:6*num_time_element,:);

error_mean_std = nan(length_vector_number,2);
error_min_max = nan(length_vector_number,2);

for i = 1:length_vector_number
    
    files_E_X_pred = dir(fullfile([Data_path, '*E_X_pred_',num2str(num_time_element*i),'_*']));
    %%%
    names_temp = {files_E_X_pred.name};

    bad = contains(names_temp,'100.mat') | contains(names_temp,'200.mat') | ...
        contains(names_temp,'400.mat') | contains(names_temp,'800.mat');

    files_E_X_pred = files_E_X_pred(~bad);
    %%%

    files_number = length(files_E_X_pred);
    errorl = zeros(files_number,1);

    for k = 1:files_number
        filePath = fullfile(Data_path, files_E_X_pred(k).name);
        load(filePath);
        commandstr_error_mse_data = ['errorl(k) = ','mean(sum((','E_X1(1:i*num_time_element,:)', '-', [files_E_X_pred(k).name(1:end-4)],').^2))/','mean(sum(E_X1.^2))',';'];
        eval(commandstr_error_mse_data);
    end
    error_mean_std(i,1) = nanmean(errorl);
    error_mean_std(i,2) = nanstd(errorl);
    error_min_max(i,1) = min(errorl);
    error_min_max(i,2) = max(errorl);
end
m1 = error_mean_std(:,1);   
s1 = error_mean_std(:,2);
neg1 = m1 - error_min_max(:,1);
pos1 = error_min_max(:,2) - m1;


a = 0;
b = 1.6e-5;
c = 1.5;
time_length = linspace(num_time_element,num_time_element*6,50)*dt;
time_length_sample = [1:6]*num_time_element*dt;
error_ana = a+b*time_length.^(c);

figure(21)
errorbar(time_length_sample, m1, neg1, pos1, 'r*','LineWidth', 1);
hold on
plot(time_length,error_ana,'k-');
xlabel('$T(\mathrm{s})$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{E,2}$', 'Interpreter', 'latex');
legend('Numerical results',['$\tilde{\varepsilon}_{E,2} = ',num2str(b*100000),'\times10^{-5}','T^{',num2str(c),'}$'], 'Interpreter', 'latex','Location','northwest');
set(gca,'fontsize',15);
set(gca,'FontName','Times New Roman');
set(gcf, 'Position', [100 100 550 400]);           
set(gcf, 'PaperPosition', [0 0 5.5 4]);            
set(gcf, 'PaperSize', [5.5 4]);
set(gca,'XScale','linear','YScale','linear');
hold off
% exportgraphics(gcf,'Fig_3_Relative_error_of_U_e_to_EX_5.pdf','Resolution',300);
% savefig('Fig_3_Relative_error_of_U_e_to_EX_5.fig');

%%%%%%%%%% Squared Wasserstein-1 distance with increasing T %%%%%%%%%%
seed_vector = [1228, 1328, 1428, 1528];
W1sq_data_E = nan(length_vector_number, 4);   % rows: T_i, cols: seeds

for i = 1:length_vector_number
    num_time_i = num_time_element * i;                 % T_i in time steps
    y_tr_sorted = sort(E_X1(1:num_time_i, :), 2);      % sort along sample dim

    for k = 1:4
        varName = ['E_X_pred_', num2str(num_time_i), '_', num2str(seed_vector(k))];
        if exist(varName, 'var')
            E_pred       = eval(varName);                  % [num_time_i x num_sample]
            y_pr_sorted  = sort(E_pred, 2);
            % W_1 at each time point, then average W_1^2 over [0, T_i]
            W1_per_t            = mean(abs(y_tr_sorted - y_pr_sorted), 2);
            W1sq_data_E(i, k)   = mean(W1_per_t .^ 2);
        end
    end
end

mu_E  = nanmean(W1sq_data_E, 2);
max_E = max(W1sq_data_E, [], 2, 'omitnan');
min_E = min(W1sq_data_E, [], 2, 'omitnan');
pos_E = max_E - mu_E;
neg_E = mu_E  - min_E;

% Reference power-law line: W_1^2 ~ T^{1.5}, consistent with the leading
% T^{1.5} term in the theoretical bounds derived from Eqs.(3.19)-(3.21).
a_W = 0;
b_W = 1.5e-5;     % tune to match numerical scale after running
c_W = 1.5;
T_ref_dense = linspace(num_time_element, num_time_element*6, 100) * dt;
T_sample    = (1:6) * num_time_element * dt;
W1sq_ana    = a_W + b_W * T_ref_dense .^ c_W;

figure(22)
errorbar(T_sample, mu_E, neg_E, pos_E, 'r*', 'LineWidth', 1);
hold on
plot(T_ref_dense, W1sq_ana, 'k-');
xlabel('$T(\mathrm{s})$', 'Interpreter', 'latex');
ylabel('$\bar{W}_{1,E}^{2}$', 'Interpreter', 'latex');
legend('Numerical results', ...
       ['$\bar{W}_{1,E}^{2} = ', ...
        num2str(b_W*1e5),'\times10^{-5}','T^{',num2str(c_W),'}$'], ...
       'Interpreter','latex','Location','northwest');
set(gca,'fontsize',15);
set(gca,'FontName','Times New Roman');
set(gcf, 'Position', [100 100 550 400]);
set(gcf, 'PaperPosition', [0 0 5.5 4]);
set(gcf, 'PaperSize', [5.5 4]);
set(gca,'XScale','linear','YScale','linear');
hold off
% exportgraphics(gcf,'Fig_4__W1sq_E_vs_T.pdf','Resolution',300);
savefig('Fig_4__W1sq_E_vs_T.fig');