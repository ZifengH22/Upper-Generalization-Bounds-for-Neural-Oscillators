clear;clc;close all;
current_path = cd;
Data_path = [current_path,'\data\'];
Data_path_sample_100 = [current_path,'\Data_path_sample_100_L1\'];
Data_path_sample_200 = [current_path,'\Data_path_sample_200_L1\'];
Data_path_sample_400 = [current_path,'\Data_path_sample_400_L1\'];
Data_path_sample_800 = [current_path,'\Data_path_sample_800_L1\'];
Data_path_sample_1600 = [current_path,'\Data_path_sample_1600_L1\'];
Data_path_sample_3200 = [current_path,'\Data_path_sample_3200_L1\'];

seed = 1328; %[1028,1128,1228,1328];
Sample_size_vector = [100,200,400,800,1600,3200];
Sample_size_number = length(Sample_size_vector);
rng(seed);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load([Data_path,'Acc.mat']);
load([Data_path,'X1_response.mat']);
% load([Data_path,'E_X1_response.mat']);

X_l = X1;
[num_time,num_sample] = size(X_l);
dt = 0.01;
t = 0:dt:(num_time-1)*dt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Acc_train = Acc;
X_train = X_l;

num_sample_select_all = Sample_size_vector(end)*1.25;
index_rand = randperm(50000, num_sample_select_all);
%%
length(unique(index_rand)) - length(index_rand)
%%

F_train_all = zeros(num_sample_select_all,num_time,1);
X_dX_input_train_all = zeros(num_sample_select_all,1,2);
X_dX_output_train_all = zeros(num_sample_select_all,num_time,1);
% E_X_output_train_all = zeros(num_sample_select_all,num_time,1);

for i = 1:num_sample_select_all
    indexl = index_rand(i);
    F_train_all(i,:,:) = Acc_train(:, indexl);
    X_dX_output_train_all(i,:,:) = X_train(:,indexl);
end
t_train_all = repmat(t,num_sample_select_all,1,1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%saving data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:Sample_size_number
    temp_position = ['Data_path_sample_',num2str(round(Sample_size_vector(i))),'_L1\data\'];
    mkdir(temp_position)
    % cd(temp_position)
    Sample_size = Sample_size_vector(i);
    num_sample_select = Sample_size*1.25;
    disp(num2str(Sample_size));

    
    F_train = F_train_all(1:num_sample_select,:,:);
    X_dX_output_train = X_dX_output_train_all(1:num_sample_select,:,:);
    X_dX_input_train = X_dX_input_train_all(1:num_sample_select,:,:);
    % E_X_output_train = E_X_output_train_all(1:num_sample_select,:,:);
    t_train = t_train_all(1:num_sample_select,:,:);

    save([temp_position,'t_train.mat'],'t_train', '-v7.3');
    save([temp_position,'F_train_',num2str(seed),'.mat'],'F_train', '-v7.3');
    save([temp_position,'X_dX_input_train.mat'],'X_dX_input_train', '-v7.3');
    save([temp_position,'X_dX_output_train_',num2str(seed),'.mat'],'X_dX_output_train', '-v7.3');
    % save([temp_position,'E_X_output_train_',num2str(seed),'.mat'],'E_X_output_train', '-v7.3');

    save([temp_position,'num_sample_select.mat'],'num_sample_select');
    save([temp_position,'Sample_size.mat'],'Sample_size');
end

disp([' '])
disp(['The number of trained whole time series is: ',num2str(num_sample_select_all)])
disp([' '])
disp(['The total length of each trained sample is: ',num2str(num_time)])


