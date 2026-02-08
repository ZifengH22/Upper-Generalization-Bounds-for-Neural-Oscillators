clear;clc;close all;
current_path = cd;
Data_path = [current_path,'\data\'];

load([Data_path,'X_train.mat']);
% load([Data_path,'Acc_train.mat']);
% load([Data_path,'X_dX_output_train.mat']);
load([Data_path,'X_pred.mat']);
load([Data_path,'coef_X_output.mat']);

load([Data_path,'Train_epochs_loss_1.mat']);
load([Data_path,'Train_epochs_loss_2.mat']);
load([Data_path,'Train_epochs_loss.mat']);
load([Data_path,'Val_epochs_loss.mat']);

[num_time,num_sample] = size(X_train);
dt = 0.01;
t = 0:dt:(num_time-1)*dt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(11)
semilogy(Train_epochs_loss(:,1),Train_epochs_loss(:,2),'k-',Train_epochs_loss_1(:,1),Train_epochs_loss_1(:,2),'m-.',Train_epochs_loss_2(:,1),Train_epochs_loss_2(:,2),'r--',Val_epochs_loss(:,1),Val_epochs_loss(:,2),'b-.');
legend('Training loss','Training loss_1','Training loss_2','Validation loss')
xlim([0,1000]);
ylim([1e-2,1e2]);
xlabel('Epoch');
ylabel('Loss');
set(gca,'fontsize',12);

%%%
error_mse = mean(mean((X_train - X_pred).^2))/mean(mean((X_train).^2));
% error_max = max(max(abs(X_train - X_pred)))/max(max(abs(X_train)))
disp(['error_mse is: ',num2str(error_mse)]);

