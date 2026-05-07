clear;clc;close all;

%%%%%%%%%%%loading data%%%%%%%%%%%
current_path = cd;
Data_path = [current_path,'\Case_1_\data\'];
Sample_number_vector = [100,200,400,800,1600,3200];
seed_vector = [1028,1128,1228,1328];
Sample_number_vector_length = length(Sample_number_vector);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file_name_l_data_X_train = [Data_path,'X1_response','.mat'];
load(file_name_l_data_X_train);
[num_time,num_sample] = size(X1);
dt = 0.01;
t = 0:dt:(num_time-1)*dt;

error_mean_std = nan(Sample_number_vector_length,2);
error_mean_std_L1 = nan(Sample_number_vector_length,2);
error_min_max = nan(Sample_number_vector_length,2);
error_min_max_L1 = nan(Sample_number_vector_length,2);

for i = 1:length(Sample_number_vector)
    
    files_X_pred = dir(fullfile([Data_path, '*X_pred_',num2str(Sample_number_vector(i)),'_*']));
    files_number = length(files_X_pred);
    errorl = [];
    errorl_L1 = [];

    for k = 1:files_number
        filePath = fullfile(Data_path, files_X_pred(k).name);
        load(filePath);
        if contains(files_X_pred(k).name, 'L1')
            commandstr_error_mse_data = ['errorl_L1 = [errorl_L1,','mean(sum((','X1', '-', [files_X_pred(k).name(1:end-4)],').^2))/','mean(sum(X1.^2))','];'];
            eval(commandstr_error_mse_data);
        else
            commandstr_error_mse_data = ['errorl = [errorl,','mean(sum((','X1', '-', [files_X_pred(k).name(1:end-4)],').^2))/','mean(sum(X1.^2))','];'];
            eval(commandstr_error_mse_data);
        end
    end
    error_mean_std(i,1) = mean(errorl);
    error_mean_std(i,2) = std(errorl);
    error_mean_std_L1(i,1) = mean(errorl_L1);
    error_mean_std_L1(i,2) = std(errorl_L1);
    if ~isempty(errorl)
        error_min_max(i,1) = min(errorl);
        error_min_max(i,2) = max(errorl);
    end
    if ~isempty(errorl_L1)
        error_min_max_L1(i,1) = min(errorl_L1);
        error_min_max_L1(i,2) = max(errorl_L1);
    end
end

%%%%%%%%%%%%%%plot results%%%%%%%%%%%%%
a_1 = 0;
b_1 = 80000;
c_1 = -2.8;
a_2 = a_1;
b_2 = 0.08;
c_2 = -0.5;
x_sample = linspace(Sample_number_vector(1),Sample_number_vector(end),200);
error_mse_ana_1 = a_1+b_1*x_sample.^(c_1);
error_mse_ana_2 = a_2+b_2*x_sample.^(c_2);
error_mse_ana = max(error_mse_ana_1,error_mse_ana_2);
% diff = error_mse_ana_1 - error_mse_ana_2;
% [~,ind_smallest] = min(diff(diff > 0));
[~,ind_smallest] = min(abs(error_mse_ana_1 - error_mse_ana_2));
error_mse_ana_1 = error_mse_ana_1(1:ind_smallest);
x_sample_1 = x_sample(1:ind_smallest);
error_mse_ana_2 = error_mse_ana_2(ind_smallest:end);
x_sample_2 = x_sample(ind_smallest:end);

m1 = error_mean_std(:,1);   
s1 = error_mean_std(:,2);
% neg1 = m1 .* s1 ./ (m1 + s1);
% pos1 = s1;
neg1 = m1 - error_min_max(:,1);
pos1 = error_min_max(:,2) - m1;

m2 = error_mean_std_L1(:,1); 
s2 = error_mean_std_L1(:,2);
% neg2 = m2 .* s2 ./ (m2 + s2);
% pos2 = s2;
neg2 = m2 - error_min_max_L1(:,1);
pos2 = error_min_max_L1(:,2) - m2;

figure(11)
errorbar(Sample_number_vector, m1, neg1, pos1, 'r*','LineWidth', 1);
hold on
errorbar(Sample_number_vector, m2, neg2, pos2, 'bo','LineWidth', 1);
plot(x_sample_1,error_mse_ana_1,'r--',x_sample_2,error_mse_ana_2,'b-');
xlabel('$N$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{X,2}$', 'Interpreter', 'latex');
legend('Numerical results from $\hat\ell_{0}$','Numerical results from $\hat\ell_{0.002}$',...
        ['$\tilde{\varepsilon}_{X,2} = ',num2str(b_1),'N^{',num2str(c_1),'}$'],...
        ['$\tilde{\varepsilon}_{X,2} = ',num2str(b_2),'N^{',num2str(c_2),'}$'], 'Interpreter', 'latex');

ylim([0.001,0.4])
xlim([100,3200])
set(gca, 'XTick', [100, 200, 400, 800, 1600, 3200]);
% set(gca,'YTick',[0.001,0.01,0.1,0.4]);
set(gca,'fontsize',15);
set(gca,'FontName','Times New Roman');
set(gcf, 'Position', [100 100 550 400]);           
set(gcf, 'PaperPosition', [0 0 5.5 4]);            
set(gcf, 'PaperSize', [5.5 4]);
set(gca,'XScale','log','YScale','log');
hold off
% exportgraphics(gcf,'Fig_1_Relative_error_of_U_e_to_X_5.pdf','Resolution',300);
% savefig('Fig_1_Relative_error_of_U_e_to_X_5.fig');

%%%%%%%%%%%PDF errors with sample size%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% Squared W_1 between push-forward of f_N and law of Y %%%%%%%%%%
% Standard 1-Wasserstein distance (Kantorovich-Rubinstein), 1-D, equal-size
% empirical samples -> exact via sort:
%   W_1 = (1/N) sum_i | y_(i) - y'_(i) |
% Plot W_1^2 so the slow-rate Rademacher prediction on MSE -- N^{-1/2} --
% transfers directly via:
%   W_1^2  <=  ( E_z|f_N - f*| )^2  <=  E_z|f_N - f*|^2  =  MSE.

W1sq_time_length = 1000;
W1sq_data    = nan(length(Sample_number_vector), 4);   % lambda_L = 0
W1sq_data_L1 = nan(length(Sample_number_vector), 4);   % lambda_L = 0.002

% Time indices used for the W_1 evaluation (same as the original loop)
t_idx = num_time - (0:W1sq_time_length-1) * (num_time / W1sq_time_length);
y_tr_block = sort(X1(t_idx, :), 2);                    % [W1sq_time_length x num_sample]

for i = 1:length(Sample_number_vector)
    for k = 1:4
        varName1 = ['X_pred_', num2str(Sample_number_vector(i)), '_', num2str(seed_vector(k))];
        varName2 = ['X_pred_', num2str(Sample_number_vector(i)), '_', num2str(seed_vector(k)), '_L1'];

        if exist(varName1, 'var') && exist(varName2, 'var')
            fX_pred   = eval(varName1);
            fX_predL1 = eval(varName2);

            y_pr_block  = sort(fX_pred  (t_idx, :), 2);
            y_prL_block = sort(fX_predL1(t_idx, :), 2);

            % W_1 at each time point, square, then average over [0, T]
            W1_per_t    = mean(abs(y_tr_block - y_pr_block ), 2);
            W1_L1_per_t = mean(abs(y_tr_block - y_prL_block), 2);

            W1sq_data   (i, k) = mean(W1_per_t   .^ 2);
            W1sq_data_L1(i, k) = mean(W1_L1_per_t.^ 2);
        end
    end
end

% Reference line: slope -1/2, constant fit in log-space
c_1 = -2.8;
b_1 = 35000;
c_2 = -0.5;
b_2 = 0.035;
x_ref    = linspace(Sample_number_vector(1), Sample_number_vector(end), 200);
ref_line_1 = b_1 * x_ref.^c_1;
ref_line_2 = b_2 * x_ref.^c_2;
ref_lines = max(ref_line_1,ref_line_2);
[~,ind_smallest] = min(abs(ref_line_1 - ref_line_2));
ref_line_1 = ref_line_1(1:ind_smallest);
x_ref_1 = x_ref(1:ind_smallest);
ref_line_2 = ref_line_2(ind_smallest:end);
x_ref_2 = x_ref(ind_smallest:end);

mu1_vector = W1sq_data;
mu1 = nanmean(mu1_vector,2);
sd1 = nanstd(mu1_vector')';
max1 = max(mu1_vector')';
min1 = min(mu1_vector')';
pos1 = max1 - mu1;
neg1 = mu1 -min1;

mu2_vector = W1sq_data_L1;
mu2 = nanmean(mu2_vector,2);
sd2 = nanstd(mu2_vector')';
max2 = max(mu2_vector')';
min2 = min(mu2_vector')';
pos2 = max2 - mu2;
neg2 = mu2 -min2;

figure(12)
errorbar(Sample_number_vector, mu1, neg1, pos1, 'r*','LineWidth',1);
hold on
errorbar(Sample_number_vector, mu2, neg2, pos2, 'bo','LineWidth',1);
plot(x_ref_1,ref_line_1,'r--',x_ref_2,ref_line_2,'b-');

xlabel('$N$', 'Interpreter','latex');
ylabel('$\bar{W}_{1,X}^{2}$', ...
       'Interpreter','latex');
legend('Numerical results from $\hat\ell_{0}$','Numerical results from $\hat\ell_{0.002}$',...
        ['$\bar{W}_{1,X}^{2} = ',num2str(b_1),'N^{',num2str(c_1),'}$'],...
        ['$\bar{W}_{1,X}^{2} = ',num2str(b_2),'N^{',num2str(c_2),'}$'], 'Interpreter', 'latex');
ylim([0.00008,2e-1]);
set(gca,'XTick',[100 200 400 800 1600 3200]);
set(gca,'fontsize',15);
set(gca,'FontName','Times New Roman');
set(gcf,'Position',     [100 100 550 400]);
set(gcf,'PaperPosition',[0 0 5.5 4]);
set(gcf,'PaperSize',    [5.5 4]);
set(gca,'XScale','log','YScale','log');
hold off
% exportgraphics(gcf,'Fig_2__W1sq_X_vs_N.pdf','Resolution',300);
% savefig('Fig_2__W1sq_X_vs_N.fig');


%%%%%%%%%% Error of extreme value CDF with N %%%%%%%%%%
W1sq_data_max    = nan(length(Sample_number_vector),4);   % lambda_L = 0
W1sq_data_max_L1 = nan(length(Sample_number_vector),4);   % lambda_L = 0.003

for i = 1:length(Sample_number_vector)
    for k = 1:4
         varName1 = ['X_pred_', num2str(Sample_number_vector(i)), '_', num2str(seed_vector(k))];
         varName2 = ['X_pred_', num2str(Sample_number_vector(i)), '_', num2str(seed_vector(k)), '_L1'];

         if exist(varName1, 'var')&exist(varName2, 'var')
             fX_pred   = eval(varName1);
             fX_predL1 = eval(varName2);

             y_tr = max(abs(X1)).';
             y_pr = max(abs(fX_pred)).';
             y_prL = max(abs(fX_predL1)).';

             % Empirical W_1 via order statistics (exact for equal-size 1-D samples)
             W1    = mean( abs( sort(y_tr) - sort(y_pr ) ) );
             W1_L1 = mean( abs( sort(y_tr) - sort(y_prL) ) );
             W1sq_data_max(i,k)    = W1   ^2;
             W1sq_data_max_L1(i,k) = W1_L1^2;
         end    
    end
   
end

% Reference line: slope -1/2, constant fit in log-space
c_1 = -2.7;
b_1 = 15000;
c_2 = -0.5;
b_2 = 0.025;
x_ref    = linspace(Sample_number_vector(1), Sample_number_vector(end), 100);
ref_line_1 = b_1 * x_ref.^c_1;
ref_line_2 = b_2 * x_ref.^c_2;
ref_lines = max(ref_line_1,ref_line_2);
[~,ind_smallest] = min(abs(ref_line_1 - ref_line_2));
ref_line_1 = ref_line_1(1:ind_smallest);
x_ref_1 = x_ref(1:ind_smallest);
ref_line_2 = ref_line_2(ind_smallest:end);
x_ref_2 = x_ref(ind_smallest:end);

mu1 = nanmean(W1sq_data_max,2);
sd1 = nanstd(W1sq_data_max')';
max1 = max(mu1_vector')';
min1 = min(mu1_vector')';
pos1 = max1 - mu1;
neg1 = mu1 -min1;

mu2 = nanmean(W1sq_data_max_L1,2);
sd2 = nanstd(W1sq_data_max_L1')';
max2 = max(mu2_vector')';
min2 = min(mu2_vector')';
pos2 = max2 - mu2;
neg2 = mu2 -min2;

figure(13)
errorbar(Sample_number_vector, mu1, neg1, pos1, 'r*','LineWidth',1);
hold on
errorbar(Sample_number_vector, mu2, neg2, pos2, 'bo','LineWidth',1);
plot(x_ref_1,ref_line_1,'r--',x_ref_2,ref_line_2,'b-');
xlabel('$N$', 'Interpreter','latex');
ylabel('$\bar{W}_{1,E}^{2}\left(\rho_{E_{X_5}},\rho_{E_{\tilde{X}_5}}\right)$', ...
       'Interpreter','latex');
legend('Numerical results from $\hat\ell_{0}$','Numerical results from $\hat\ell_{0.002}$',...
        ['$\bar{W}_{1,E}^{2}\left(\rho_{E_{X_5}},\rho_{E_{\tilde{X}_5}}\right) = ',num2str(b_1),'N^{',num2str(c_1),'}$'],...
        ['$\bar{W}_{1,E}^{2}\left(\rho_{E_{X_5}},\rho_{E_{\tilde{X}_5}}\right) = ',num2str(b_2),'N^{',num2str(c_2),'}$'], 'Interpreter', 'latex');
% ylim([0.0002,0.02]);
set(gca,'XTick',[100 200 400 800 1600 3200]);
set(gca,'fontsize',15);
set(gca,'FontName','Times New Roman');
set(gcf,'Position',     [100 100 550 400]);
set(gcf,'PaperPosition',[0 0 5.5 4]);
set(gcf,'PaperSize',    [5.5 4]);
set(gca,'XScale','log','YScale','log');
hold off
% exportgraphics(gcf,'Fig_3__W1sq_E_vs_N.pdf','Resolution',300);
% savefig('Fig_3__W1sq_E_vs_N.fig');
