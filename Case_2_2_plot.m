clear;clc;close all;

%%%%%%%%%%%loading data%%%%%%%%%%%
current_path = cd;
Data_path = [current_path,'\Case_2_\data\'];

length_vector = [500,1000,1500,2000,2500,3000];
length_vector_number = length(length_vector);
Sample_vector = [100,200,400,800,1600];
Sample_vector_number = length(Sample_vector);

%%%%loading data%%%%%%%%
num_time_element = 500;
file_name_l_data_X_train = [Data_path,'E_X1_response','.mat'];
load(file_name_l_data_X_train);
[num_time,num_sample] = size(E_X1);
dt = 0.01;
t = 0:dt:(num_time-1)*dt;
E_X1 = E_X1(1:6*num_time_element,:);

files_E_X_pred = dir(fullfile([Data_path, '*E_X_pred_',num2str(length_vector(end)),'_*']));
files_number = length(files_E_X_pred);
for k = 1:files_number
    filePath = fullfile(Data_path, files_E_X_pred(k).name);
    load(filePath); 
end

%%%%%%%%%PDF and CDF of EX%%%%%%%%%%%%

%%%%%%%%%PDF%%%%%%%%%
E_X_range_PDF = linspace(1,33,100);
bandwidth = 0.1;

[PDF_E_X1,~] = ksdensity(E_X1(end,:),E_X_range_PDF,'Bandwidth',bandwidth);
[PDF_E_X_pred_100,~] = ksdensity(E_X_pred_3000_1228_100(end,:),E_X_range_PDF,'Bandwidth',bandwidth);
[PDF_E_X_pred_200,~] = ksdensity(E_X_pred_3000_1228_200(end,:),E_X_range_PDF,'Bandwidth',bandwidth);
[PDF_E_X_pred_400,~] = ksdensity(E_X_pred_3000_1228_400(end,:),E_X_range_PDF,'Bandwidth',bandwidth);
[PDF_E_X_pred_800,~] = ksdensity(E_X_pred_3000_1228_800(end,:),E_X_range_PDF,'Bandwidth',bandwidth);
[PDF_E_X_pred_1600,~] = ksdensity(E_X_pred_3000_1228(end,:),E_X_range_PDF,'Bandwidth',bandwidth);

LineWidth_PDF = 1;
MarkerSize_PDF = 4;
figure(23)

plot(E_X_range_PDF,PDF_E_X_pred_100 ,'-o', ...
    'Color',[0 0.4470 0.7410],'LineWidth',0.5*LineWidth_PDF,'MarkerSize',MarkerSize_PDF)
hold on

plot(E_X_range_PDF,PDF_E_X_pred_200 ,'-s', ...
    'Color',[0.8500 0.3250 0.0980],'LineWidth',0.5*LineWidth_PDF,'MarkerSize',MarkerSize_PDF)

plot(E_X_range_PDF,PDF_E_X_pred_400 ,'-d', ...
    'Color',[0.4940 0.1840 0.5560],'LineWidth',0.5*LineWidth_PDF,'MarkerSize',MarkerSize_PDF)

plot(E_X_range_PDF,PDF_E_X_pred_800 ,'-^', ...
    'Color',[0.6350 0.0780 0.1840],'LineWidth',0.5*LineWidth_PDF,'MarkerSize',MarkerSize_PDF)

plot(E_X_range_PDF,PDF_E_X_pred_1600,'.-', ...
    'Color',[0.25 0.25 0.25],'LineWidth',0.5*LineWidth_PDF)

plot(E_X_range_PDF,PDF_E_X1,'k-','LineWidth',LineWidth_PDF); 

hold off

xlim([0,25]);
ylim([0,0.3]);
legend('$\tilde{E}_{X_5,l}(30)$ ($N = 100$)',...
    '$\tilde{E}_{X_5,l}(30)$ ($N = 200$)',...
    '$\tilde{E}_{X_5,l}(30)$ ($N = 400$)',...
    '$\tilde{E}_{X_5,l}(30)$ ($N = 800$)',...
    '$\tilde{E}_{X_5,l}(30)$ ($N = 1600$)',...
    '$\it{E_{X_\mathrm{5},l}(\mathrm{30})}$', 'Interpreter', 'latex');


xlabel('$\it{E_{X_\mathrm{5}}(\mathrm{30})}$', 'Interpreter', 'latex');
ylabel('PDF', 'Interpreter', 'latex');
set(gca,'fontsize',15);
set(gca,'FontName','Times New Roman');
set(gcf, 'Position', [100 100 550 400]);           
set(gcf, 'PaperPosition', [0 0 5.5 4]);            
set(gcf, 'PaperSize', [5.5 4]);
set(gca,'yscale','linear');
% savefig('Fig_7_PDF_EX_30.fig');


%%%%%%%%%CDF%%%%%%%%%
[CDF_E_X_pred_100,E_X_pred_range_CDF_100] = ecdf(E_X_pred_3000_1228_100(end,:));
[CDF_E_X_pred_200,E_X_pred_range_CDF_200] = ecdf(E_X_pred_3000_1228_200(end,:));
[CDF_E_X_pred_400,E_X_pred_range_CDF_400] = ecdf(E_X_pred_3000_1228_400(end,:));
[CDF_E_X_pred_800,E_X_pred_range_CDF_800] = ecdf(E_X_pred_3000_1228_800(end,:));
[CDF_E_X_pred_1600,E_X_pred_range_CDF_1600] = ecdf(E_X_pred_3000_1228(end,:));
[CDF_E_X1,E_X1_range_CDF] = ecdf(E_X1(end,:));
CDF_E_X1_y = CDF_E_X1;

eps = 1e-5;
CDF_E_X_pred_100 = min(max(CDF_E_X_pred_100, eps), 1 - eps);
CDF_E_X_pred_200 = min(max(CDF_E_X_pred_200, eps), 1 - eps);
CDF_E_X_pred_400 = min(max(CDF_E_X_pred_400, eps), 1 - eps);
CDF_E_X_pred_800 = min(max(CDF_E_X_pred_800, eps), 1 - eps);
CDF_E_X_pred_1600 = min(max(CDF_E_X_pred_1600, eps), 1 - eps);
CDF_E_X1 = min(max(CDF_E_X1, eps), 1 - eps);
CDF_E_X1_y = min(max(CDF_E_X1_y, eps/2), 1 - eps/2);

z_pred_100  = norminv(CDF_E_X_pred_100);
z_pred_200  = norminv(CDF_E_X_pred_200);
z_pred_400  = norminv(CDF_E_X_pred_400);
z_pred_800  = norminv(CDF_E_X_pred_800);
z_pred_1600 = norminv(CDF_E_X_pred_1600);
z_1   = norminv(CDF_E_X1);
z_1_y = norminv(CDF_E_X1_y);

z_1_y_5 = linspace(z_1_y(1), z_1_y(end), 5);
idx = zeros(size(z_1_y_5));
for i = 1:length(z_1_y_5)
    [~, idx(i)] = min(abs(z_1_y - z_1_y_5(i)));
end
prob_labels = CDF_E_X1_y(idx);

MarkerSize_CDF = 8;
figure(24)
plot(E_X1_range_CDF,         z_1,         'k*', ...
     'MarkerSize', MarkerSize_CDF);
hold on
plot(E_X_pred_range_CDF_100,  z_pred_100,  'o',  ...
     E_X_pred_range_CDF_200,  z_pred_200,  's',  ...
     E_X_pred_range_CDF_400,  z_pred_400,  'd',  ...
     E_X_pred_range_CDF_800,  z_pred_800,  '^',  ...
     E_X_pred_range_CDF_1600, z_pred_1600, 'v',  ...
     'MarkerSize', MarkerSize_CDF*0.5);
hold off

xlim([0,35]);
legend('$\it{E_{X_\mathrm{5},l}(\mathrm{30})}$', ...
    '$\tilde{E}_{X_5,l}(30)$ ($N = 100$)',...
    '$\tilde{E}_{X_5,l}(30)$ ($N = 200$)',...
    '$\tilde{E}_{X_5,l}(30)$ ($N = 400$)',...
    '$\tilde{E}_{X_5,l}(30)$ ($N = 800$)',...
    '$\tilde{E}_{X_5,l}(30)$ ($N = 1600$)',...
    'Interpreter', 'latex','location','southeast');

xlabel('$\it{E_{X_\mathrm{5}}(\mathrm{30})}$', 'Interpreter', 'latex');
ylabel('CDF', 'Interpreter', 'latex');
yticks(z_1_y_5);
yticklabels(compose('%.3f', prob_labels));
ylim([min(z_1_y), max(z_1_y)]);
set(gca,'fontsize',15);
set(gca,'FontName','Times New Roman');
set(gcf, 'Position', [100 100 550 400]);
set(gcf, 'PaperPosition', [0 0 5.5 4]);
set(gcf, 'PaperSize', [5.5 4]);
set(gca, 'xscale', 'linear', 'yscale', 'linear');
% savefig('Fig_5_CDF_EX_30.fig');

%%%%%%%%%W-1 distance%%%%%%%%%
W1_100 = mean( abs( sort(E_X_pred_3000_1228_100(end,:)) - sort(E_X1(end,:)) ) );
W1_200 = mean( abs( sort(E_X_pred_3000_1228_200(end,:)) - sort(E_X1(end,:)) ) );
W1_400 = mean( abs( sort(E_X_pred_3000_1228_400(end,:)) - sort(E_X1(end,:)) ) );
W1_800 = mean( abs( sort(E_X_pred_3000_1228_800(end,:)) - sort(E_X1(end,:)) ) );
W1_1600 = mean( abs( sort(E_X_pred_3000_1228(end,:)) - sort(E_X1(end,:)) ) );
W_sqr_vector = [W1_100,W1_200,W1_400,W1_800,W1_1600].^2; 

c_ref = -1;
b_ref = 25;
x_ref    = linspace(Sample_vector(1), Sample_vector(end), 50);
ref_line = b_ref * x_ref.^c_ref;

figure(25)
plot(Sample_vector, W_sqr_vector, 'r*', 'LineWidth', 1);
hold on
plot(x_ref, ref_line, 'k-');
xlabel('$N$', 'Interpreter','latex');
ylabel('$W_{1,E_{X_5}(30)}^{2}$', ...
       'Interpreter','latex');
legend('Numerical results',['$W_{1,E_{X_5}(30)}^{2} = ',num2str(b_ref),'N^{',num2str(c_ref),'}$'], 'Interpreter', 'latex','Location','northeast');
       
ylim([0.001,1]);
set(gca,'XTick',[100 200 400 800 1600 3200]);
set(gca,'fontsize',15);
set(gca,'FontName','Times New Roman');
set(gcf,'Position',     [100 100 550 400]);
set(gcf,'PaperPosition',[0 0 5.5 4]);
set(gcf,'PaperSize',    [5.5 4]);
set(gca,'XScale','log','YScale','log');
hold off
% savefig('Fig_6__W1sq_E30_vs_N.fig');



