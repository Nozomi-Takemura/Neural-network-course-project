%% Load the choles data
%a
% This will create a 21x264 choInputs matrix of 264 input patterns
% and a 3x264 matrix choTargets of output patterns
doc cho_dataset
load cho_dataset
%% Standardize the variables
[pn, std_p] = mapstd(choInputs);
[tn, std_t] = mapstd(choTargets);
%% check the correlation matrix before pca
cormat = corr(pn') ;
	
set(gca, 'xtick', 1:21);
set(gca, 'ytick', 1:21);
colormap(gray);
colorbar;
%% PCA
[pp, pca_p] = processpca(pn, 'maxfrac', 0.001);
[m, n] = size(pp)
[coeff,score,latent] = pca(choInputs);
%% correlation matrix after pca
cormat = corr(pp') ;
imagesc(cormat);
set(gca, 'xtick', 1:4);
set(gca, 'ytick', 1:4);
colormap(gray);
colorbar;
%% b
%% Set indices for test, validation and training sets
Test_ix = 2:4:n;
Val_ix = 4:4:n;
Train_ix = [1:4:n 3:4:n];
%% Configure a network using LM
net = fitnet(5);
net.divideFcn = 'divideind';
net.divideParam = struct('trainInd', Train_ix, ...
'valInd', Val_ix, ...
'testInd', Test_ix);
[net, tr] = train(net, pn, tn);
%% Get predictions on training and test
Yhat_train = net(pn(:, Train_ix));
Yhat_test = net(pn(:, Test_ix));
MSE_test = perform(net,tn(:,Test_ix),Yhat_test)
MSE_train = perform(net,tn(:,Train_ix),Yhat_train)
%% Configure a network using bayesian regularization
net2 = fitnet(5);
net2.trainFCn= 'trainbr';
net2.divideFcn = 'divideind';
net2.divideParam = struct('trainInd', Train_ix, ...
'valInd', Val_ix, ...
'testInd', Test_ix);
[net2, tr2] = train(net2, pn, tn);
%% Get predictions on training and test
Yhat_train2 = net2(pn(:, Train_ix));
Yhat_test2 = net2(pn(:, Test_ix));
MSE_test2 = perform(net2,tn(:,Test_ix),Yhat_test2)
MSE_train2 = perform(net2,tn(:,Train_ix),Yhat_train2)

%% Configure a network using LM (reduced input)
net3 = fitnet(5);
net3.divideFcn = 'divideind';
net3.divideParam = struct('trainInd', Train_ix, ...
'valInd', Val_ix, ...
'testInd', Test_ix);
[net3, tr3] = train(net3, pp, tn);
%% Get predictions on training and test
Yhat_train3 = net3(pp(:, Train_ix));
Yhat_test3 = net3(pp(:, Test_ix));
MSE_test3 = perform(net3,tn(:,Test_ix),Yhat_test3)
MSE_train3 = perform(net3,tn(:,Train_ix),Yhat_train3)
%% Configure a network using bayesian regularization (reduced input)
net4 = fitnet(5);
net4.trainFCn= 'trainbr';
net4.divideFcn = 'divideind';
net4.divideParam = struct('trainInd', Train_ix, ...
'valInd', Val_ix, ...
'testInd', Test_ix);
[net4, tr4] = train(net4, pp, tn);
%% Get predictions on training and test
Yhat_train4 = net4(pp(:, Train_ix));
Yhat_test4 = net4(pp(:, Test_ix));
MSE_test4 = perform(net4,tn(:,Test_ix),Yhat_test4)
MSE_train3 = perform(net3,tn(:,Train_ix),Yhat_train4)

%% Input selection by ARD (demard)
addpath('/home/nozomi/package/netlab');
demard
%% demev1 demo
demev1
%% UCI data
clear;
uci = importdata('/home/nozomi/Documents/2017courses/DMNN/Ex/Data/ionstart.mat');
inputs = uci.Xnorm;
targets = hardlim(uci.Y); 
%% training and test sets
rp = randperm(351);
Test_ix = rp(1:100);
Train_ix = setdiff(1:351, Test_ix); %returns values of A not in B
%% setting up network (m = 33)
nin = 33;
nhidden = 5;
nout = 1;
aw1 = 0.01*ones(1, nin);
ab1 = 0.01;
aw2 = 0.01;
ab2 = 0.01;
prior = mlpprior(nin, nhidden, nout, aw1, ab1, aw2, ab2);
net = mlp (nin, nhidden, nout, 'logistic', prior); %beta not needed

% Set up vector of options for the optimiser.
nouter = 2;			% Number of outer loops
ninner = 10;		        % Number of inner loops
options = zeros(1,18);		% Default options vector.
options(1) = 1;			% This provides display of error values.
options(2) = 1.0e-7;	% This ensures that convergence must occur
options(3) = 1.0e-7;
options(14) = 300;		% Number of training cycles in inner loop/max number of function evaluations

%% Train using scaled conjugate gradients,
for k = 1:nouter
  net = netopt(net, options, inputs(Train_ix,:), targets(Train_ix,:), 'scg');
  [net, gamma] = evidence(net, inputs(Train_ix,:), targets(Train_ix,:), ninner);
end
  
%% 
[outputs, z] = mlpfwd(net, inputs(Test_ix,:));
figure, plotconfusion(targets(Test_ix,:)', outputs'), legend('Original Inputs');
figure, plotroc(targets(Test_ix,:)', outputs'), legend('Original Inputs');

%% 
[X,Yc,T,AUCo] = perfcurve(targets(Test_ix,:),outputs,1);
v = [outputs targets(Test_ix,:)];
classes_f = round(outputs,0);
percError_f = sum(abs(targets(Test_ix,:)-classes_f))/(length(classes_f));
%% input selection
ind = 1:33;
w = net.alpha(1:33);
w = [ind' w];
w = sortrows(w,2,'descend');
%w = sortrows(w,2);
ppp = w(:,2);
pp = w(:,1);
figure;
%bar(pp, ppp)
%xlabel('Index of input variables');
plot(1:33, log(w(:, 2)), 'ko');
ylabel('log(Hyperparameter)')
xlabel('Number of inputs')%sharp drop at 4


%% Reduced: setting up network (m = 4)
ninr = 4;
%ind = w(1:ninr, 1);
ind = w(end-ninr+1:end, 1);
inputs2 = uci.Xnorm(:, ind);
nhiddenr = 5;
noutr = 1;
aw1r = 0.01*ones(1, ninr);
ab1r = 0.01;
aw2r = 0.01;
ab2r = 0.01;
priorr = mlpprior(ninr, nhiddenr, noutr, aw1r, ab1r, aw2r, ab2r);
netr = mlp (ninr, nhiddenr, noutr, 'logistic', priorr); %beta not needed

% Set up vector of options for the optimiser.
nouterr = 2;			% Number of outer loops
ninnerr = 10;		        % Number of inner loops
options = zeros(1,18);		% Default options vector.
options(1) = 1;			% This provides display of error values.
options(2) = 1.0e-7;	% This ensures that convergence must occur
options(3) = 1.0e-7;
options(14) = 300;		% Number of training cycles in inner loop/max number of function evaluations

%% Train using scaled conjugate gradients,
for k = 1:nouterr
  netr = netopt(netr, options, inputs2(Train_ix,:), targets(Train_ix,:), 'scg');
  [netr, gammar] = evidence(netr, inputs2(Train_ix,:), targets(Train_ix,:), ninnerr);
end
  
%% 
[outputs2, zr] = mlpfwd(netr, inputs2(Test_ix,:));
figure, plotconfusion(targets(Test_ix,:)', outputs2'), legend('Reduced Inputs');
figure, plotroc(targets(Test_ix,:)', outputs2'), legend('Reduced Inputs');

%% 
[X,Yc,T,AUCr] = perfcurve(targets(Test_ix,:),outputs2,1);
vr = [outputs2 targets(Test_ix,:)];
%cutoff 0,5
classes_ard = round(outputs2,0);
percError_test = sum(abs(targets(Test_ix,:)-classes_ard))/(length(classes_ard));
