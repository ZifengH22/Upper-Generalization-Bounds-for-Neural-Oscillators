clear;clc;close all;
current_path = cd;
Data_path = [current_path,'\data\'];

load([Data_path,'Train_epochs_loss_1.mat']);
load([Data_path,'Train_epochs_loss_2.mat']);
load([Data_path,'Train_epochs_loss.mat']);
load([Data_path,'Val_epochs_loss.mat']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
semilogy(Train_epochs_loss(:,1),Train_epochs_loss(:,2),'k-',Train_epochs_loss_1(:,1),Train_epochs_loss_1(:,2),'m-.',Train_epochs_loss_2(:,1),Train_epochs_loss_2(:,2),'r--',Val_epochs_loss(:,1),Val_epochs_loss(:,2),'b-.');
legend('Training loss','Training loss_1','Training loss_2','Validation loss')
xlim([0,4000]);
ylim([1e-2,1e2]);
xlabel('Epoch');
ylabel('Loss');
set(gca,'fontsize',12);
