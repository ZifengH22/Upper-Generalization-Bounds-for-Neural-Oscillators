clear;clc;close all;
current_path = cd;
Data_path = [current_path,'\data\'];

seed = 1228; %[1228,1328,1428,1528];
rng(seed);
Sample_size = [100,200,400,800];
Sample_size_number = length(Sample_size);
time_length = 3000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load([Data_path,'Acc.mat']);
load([Data_path,'X1_response.mat']);
load([Data_path,'E_X1_response.mat']);
Acc = Acc(1:time_length,:);
X1 = X1(1:time_length,:);
E_X1 = E_X1(1:time_length,:);

[num_time,num_sample] = size(X1);
dt = 0.01;
t = 0:dt:(num_time-1)*dt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Acc_train = Acc;
X_train = X1;
E_X_train = E_X1;
index_rand = randperm(50000, 1600);

for n = 1:Sample_size_number
    num_sample_select = Sample_size(n)*1.25;
    index_rand_temp = index_rand(1:num_sample_select);

    F_train = zeros(num_sample_select,num_time,1);
    X_dX_input_train = zeros(num_sample_select,1,2);
    X_dX_output_train = zeros(num_sample_select,num_time,1);
    E_X_output_train = zeros(num_sample_select,num_time,1);

    for i = 1:num_sample_select
        indexl = index_rand_temp(i);
        F_train(i,:,:) = Acc_train(:, indexl);
        X_dX_output_train(i,:,:) = X_train(:,indexl);
        E_X_output_train(i,:,:) = E_X_train(:,indexl);
    end
    t_train = repmat(t,num_sample_select,1,1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%saving data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    temp_position = ['Data_path_time_length_3000_',num2str(seed),'_',num2str(Sample_size(n)),'\data\'];
    mkdir(temp_position)
    % cd(temp_position)
    disp(num2str(n))
    save([temp_position,'t_train.mat'],'t_train', '-v7.3');
    save([temp_position,'F_train_',num2str(seed),'.mat'],'F_train', '-v7.3');
    save([temp_position,'X_dX_input_train.mat'],'X_dX_input_train', '-v7.3');
    save([temp_position,'X_dX_output_train_',num2str(seed),'.mat'],'X_dX_output_train', '-v7.3');
    save([temp_position,'E_X_output_train_',num2str(seed),'.mat'],'E_X_output_train', '-v7.3');

    save([temp_position,'num_sample_select.mat'],'num_sample_select');
    tmp.Sample_size = Sample_size(n);
    save([temp_position,'Sample_size.mat'], '-struct', 'tmp');

    disp([' '])
    disp(['The number of trained whole time series is: ',num2str(num_sample_select)])
    disp([' '])
    disp(['The total length of each trained sample is: ',num2str(num_time)])
    disp([' '])
    disp(['The total number of trained samples is: ',num2str(Sample_size(n))])

end


