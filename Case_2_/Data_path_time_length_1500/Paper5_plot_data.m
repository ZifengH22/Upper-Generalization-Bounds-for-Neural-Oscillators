clear;clc;close all;
current_path = cd;
Data_path = [current_path,'\data\'];

seed_vector = [1228,1328,1428,1528];

for i = 1:length(seed_vector)
    seed = seed_vector(i);
    load([Data_path,'Train_epochs_loss_1_',num2str(seed),'.mat']);
    load([Data_path,'Train_epochs_loss_2_',num2str(seed),'.mat']);
    load([Data_path,'Train_epochs_loss_',num2str(seed),'.mat']);
    load([Data_path,'Val_epochs_loss_',num2str(seed),'.mat']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
semilogy(Train_epochs_loss_1228(:,1),Train_epochs_loss_1228(:,2),'k-',Train_epochs_loss_1_1228(:,1),Train_epochs_loss_1_1228(:,2),'m-.',Train_epochs_loss_2_1228(:,1),Train_epochs_loss_2_1228(:,2),'r--',Val_epochs_loss_1228(:,1),Val_epochs_loss_1228(:,2),'b-.');
legend('Training loss','Training loss_1','Training loss_2','Validation loss')
xlim([0,3000]);
ylim([1e-2,1e2]);
xlabel('Epoch');
ylabel('Loss');
set(gca,'fontsize',12);

figure(2)
semilogy(Train_epochs_loss_1328(:,1),Train_epochs_loss_1328(:,2),'k-',Train_epochs_loss_1_1328(:,1),Train_epochs_loss_1_1328(:,2),'m-.',Train_epochs_loss_2_1328(:,1),Train_epochs_loss_2_1328(:,2),'r--',Val_epochs_loss_1328(:,1),Val_epochs_loss_1328(:,2),'b-.');
legend('Training loss','Training loss_1','Training loss_2','Validation loss')
xlim([0,3000]);
ylim([1e-2,1e2]);
xlabel('Epoch');
ylabel('Loss');
set(gca,'fontsize',12);

figure(3)
semilogy(Train_epochs_loss_1428(:,1),Train_epochs_loss_1428(:,2),'k-',Train_epochs_loss_1_1428(:,1),Train_epochs_loss_1_1428(:,2),'m-.',Train_epochs_loss_2_1428(:,1),Train_epochs_loss_2_1428(:,2),'r--',Val_epochs_loss_1428(:,1),Val_epochs_loss_1428(:,2),'b-.');
legend('Training loss','Training loss_1','Training loss_2','Validation loss')
xlim([0,3000]);
ylim([1e-2,1e2]);
xlabel('Epoch');
ylabel('Loss');
set(gca,'fontsize',12);

figure(4)
semilogy(Train_epochs_loss_1528(:,1),Train_epochs_loss_1528(:,2),'k-',Train_epochs_loss_1_1528(:,1),Train_epochs_loss_1_1528(:,2),'m-.',Train_epochs_loss_2_1528(:,1),Train_epochs_loss_2_1528(:,2),'r--',Val_epochs_loss_1528(:,1),Val_epochs_loss_1528(:,2),'b-.');
legend('Training loss','Training loss_1','Training loss_2','Validation loss')
xlim([0,3000]);
ylim([1e-2,1e2]);
xlabel('Epoch');
ylabel('Loss');
set(gca,'fontsize',12);


