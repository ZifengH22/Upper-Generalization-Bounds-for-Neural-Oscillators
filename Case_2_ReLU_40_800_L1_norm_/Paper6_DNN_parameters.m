clear;clc;close all;
current_path = cd;
Data_path = [current_path,'\data\'];

load([Data_path,'W_RK4GRUcell0.mat'])
load([Data_path,'W_RK4GRUcelle.mat'])
load([Data_path,'B_RK4GRUcell0.mat'])
load([Data_path,'B_RK4GRUcelle.mat'])

load([Data_path,'W_top_DNN0.mat'])
load([Data_path,'W_top_DNNe.mat'])
load([Data_path,'B_top_DNN0.mat'])
load([Data_path,'B_top_DNNe.mat'])

% Lip1 = ((sum(sum(W_RK4GRUcell0.^2))*sum(sum(W_RK4GRUcelle.^2))).^0.5+1).^2;
% Lip2 = sum(sum(W_top_DNN0.^2))*sum(sum(W_top_DNNe.^2));
% Lip = Lip1*Lip2;

Lip_1 = sum(sum(W_RK4GRUcell0.^2)) + sum(sum(B_RK4GRUcell0.^2));
Lip_2 = sum(sum(W_RK4GRUcelle.^2)) + sum(sum(B_RK4GRUcelle.^2));
Lip_3 = sum(sum(W_top_DNN0.^2)) + sum(sum(B_top_DNN0.^2));
Lip_4 = sum(sum(W_top_DNNe.^2)) + sum(sum(B_top_DNNe.^2));
Lip = ((Lip_1*Lip_2)^0.5+1)^2*Lip_3*Lip_4;

L1_1 = sum(sum(abs(W_RK4GRUcell0))) + sum(sum(abs(B_RK4GRUcell0)));
L1_2 = sum(sum(abs(W_RK4GRUcelle))) + sum(sum(abs(B_RK4GRUcelle)));
L1_3 = sum(sum(abs(W_top_DNN0))) + sum(sum(abs(B_top_DNN0)));
L1_4 = sum(sum(abs(W_top_DNNe))) + sum(sum(abs(B_top_DNNe)));
L1 = L1_1+L1_2+L1_3+L1_4;

L2_1 = sum(sum(W_RK4GRUcell0.^2)) + sum(sum(B_RK4GRUcell0.^2));
L2_2 = sum(sum(W_RK4GRUcelle.^2)) + sum(sum(B_RK4GRUcelle.^2));
L2_3 = sum(sum(W_top_DNN0.^2)) + sum(sum(B_top_DNN0.^2));
L2_4 = sum(sum(W_top_DNNe.^2)) + sum(sum(B_top_DNNe.^2));
L2 = L2_1+L2_2+L2_3+L2_4;

Lmax_1_W = max(max(abs(W_RK4GRUcell0))); 
Lmax_1_B = max(max(abs(B_RK4GRUcell0)));
Lmax_2_W = max(max(abs(W_RK4GRUcelle))); 
Lmax_2_B = max(max(abs(B_RK4GRUcelle)));

Lmax_3_W = max(max(abs(W_top_DNN0)));
Lmax_3_B = max(max(abs(B_top_DNN0)));
Lmax_4_W = max(max(abs(W_top_DNNe)));
Lmax_4_B = max(max(abs(B_top_DNNe)));

Lmax = Lmax_1_W + Lmax_1_B + Lmax_2_W + Lmax_2_B + Lmax_3_W + Lmax_3_B + Lmax_4_W + Lmax_4_B;

%%%%%%%%%%Lip_enhance%%%%%%%%%%
[m,k] = size(W_RK4GRUcelle);
[~,n] = size(W_RK4GRUcell0);

A = reshape(W_RK4GRUcelle, [m, k, 1]);  % m x k x 1
B = reshape(W_RK4GRUcell0, [1, k, n]); % 1 x k x n
prod = A .* B;

prod_relu = max(0, prod);
prod_relu_min = max(0, -prod);

C = max(sum(prod_relu, 2), sum(prod_relu_min, 2));   % m x 1 x n
C = squeeze(C); 
Lip_RK4GRUcell = sum(C(:))

[m,k] = size(W_top_DNNe);
[~,n] = size(W_top_DNN0);

A = reshape(W_top_DNNe, [m, k, 1]);  % m x k x 1
B = reshape(W_top_DNN0, [1, k, n]); % 1 x k x n
prod = A .* B;

prod_relu = max(0, prod);
prod_relu_min = max(0, -prod);

C = max(sum(prod_relu, 2), sum(prod_relu_min, 2));   % m x 1 x n
C = squeeze(C); 
Lip_top_DNN = sum(C(:))

Lip_enhance  = Lip_RK4GRUcell+ Lip_top_DNN
