%% santa fe time series
%load data
clear all

cd /home/nozomi;
traindat = importdata('/home/nozomi/Documents/2017courses/DMNN/Ex/Data/lasertrain.dat');
pred = importdata('/home/nozomi/Documents/2017courses/DMNN/Ex/Data/laserpred.dat');
cd 'Documents/2017courses/DMNN/Ex/2/';
addpath(pwd);

[a, p] = getTimeSeriesTrainData(traindat, 50);
[inputs,inputStates,layerStates,targets] = preparets(net,a,{},p);

%% plot
plot(1:length(traindat), traindat, '-k'), hold on;
plot((length(traindat)+1):(length(traindat)+length(pred)), pred, ' -r');
xlabel('Time')
ylabel('Intensity')
legend('Training Data', 'Test Data')
hold off
autocorr(traindat, 999)
autocorr(traindat, 100)
%% training and validation
Train_ix = [1:85, 101:185, 201:285, 301:385, 401:485,501:585, 601:685, 701:785, 801:885, 901:985];
Val_ix = [];
Test_ix = [86:100, 186:200, 286:300, 386:400, 486:500, 586:600, 686:700, 786:800, 886:900, 986:1000];
x_train = tonndata(traindat, false, false);
testpred = tonndata(pred, false, false);
%% network  1 LM 50 lags 10 neurons%
lags = 50;
n_neurons = 10;
net = narnet(1:lags, n_neurons);
view(net);
net.trainFcn = 'trainlm';
net.performParam.regularization = 1e-6;
net.divideFcn = 'divideind';
net.divideParam = struct('trainInd', Train_ix, ...
                         'valInd', Test_ix, ...
                         'testInd', Val_ix);

[Xs, Xi, Ai, Ts] = preparets(net, {}, {}, x_train);
%%  test nozomi-123117
lags = 50;
n_neurons = 10;
net = narnet(1:lags, n_neurons);
view(net);
net.trainFcn = 'trainlm';
net.performParam.regularization = 1e-6;
net.divideParam.testRatio = 0;
net.divideParam.trainRatio = 0.85;
net = train(net,Xs,Ts,Xi,Ai);
[Y,Xf,Af] = net(Xs,Xi,Ai); 
perf = perform(net,Ts,Y)

[netc,Xic,Aic] = closeloop(net,Xf,Af); 
view(netc)
y2 = netc(cell(0,100),Xic,Aic)
%%
net = closeloop(train(net, Xs, Ts, Xi, Ai)); 
net = closeloop(train(net, Xs, Ts, Xi, Ai)); 

[Xs, Xi, Ai, Ts] = preparets(net, {}, {}, x_train);
net = train(net,Xs,Ts,Xi,Ai);
view(net)
%% 
ypred = nan(100+lags, 1);
ypred = tonndata(ypred, false, false);
ypred(1:lags) = x_train(end-(lags-1):end);
[xc, xic, aic, tc] = preparets(net, {}, {}, ypred);
ypred = fromnndata(net(xc, xic, aic), true, false, false);
residuals = ypred - pred;
output = num2cell(ypred)';

%% performance check
MSE = perform(net, testpred, output);
MAE = mae(net, testpred, tonndata(ypred, false, false));
%% plotting it
plot (pred, 'b-');
hold on;
plot(ypred, 'g-');
hold off
legend('Test Data', 'Fitted Values')
%% network  1 LM 50 lags 15 neurons%
lags = 50;
n_neurons = 15;
net = narnet(1:lags, n_neurons);
net.trainFcn = 'trainlm';
net.performParam.regularization = 1e-6;
net.divideFcn = 'divideind';
net.divideParam = struct('trainInd', Train_ix, ...
                         'valInd', Test_ix, ...
                         'testInd', Val_ix);
[Xs, Xi, Ai, Ts] = preparets(net, {}, {}, x_train);
net = closeloop(train(net, Xs, Ts, Xi, Ai)); 

%% 
ypred = nan(100+lags, 1);
ypred = tonndata(ypred, false, false);
ypred(1:lags) = x_train(end-(lags-1):end);
[xc, xic, aic, tc] = preparets(net, {}, {}, ypred);
%converting to predictions
ypred = fromnndata(net(xc, xic, aic), true, false, false);
residuals = ypred - pred;
output = num2cell(ypred)';

%% performance check
MSE = perform(net, testpred, output);
MAE = mae(net, testpred, tonndata(ypred, false, false));
%% plotting it
plot (pred, 'b-');
hold on;
plot(ypred, 'g-');
hold off
legend('Test Data', 'Fitted Values')

%% network  1 BR 50 lags 15 neurons%
lags = 50;
n_neurons = 15;
net = narnet(1:lags, n_neurons);
net.trainFcn = 'trainbr';
net.performParam.regularization = 1e-6;
net.divideFcn = 'divideind';
net.divideParam = struct('trainInd', Train_ix, ...
                         'valInd', Test_ix, ...
                         'testInd', Val_ix);
[Xs, Xi, Ai, Ts] = preparets(net, {}, {}, x_train);
net = closeloop(train(net, Xs, Ts, Xi, Ai)); 

%% 
ypred = nan(100+lags, 1);
ypred = tonndata(ypred, false, false);
ypred(1:lags) = x_train(end-(lags-1):end);
[xc, xic, aic, tc] = preparets(net, {}, {}, ypred);
%converting to predictions
ypred = fromnndata(net(xc, xic, aic), true, false, false);
residuals = ypred - pred;
output = num2cell(ypred)';

%% performance check
MSE = perform(net, testpred, output);
MAE = mae(net, testpred, tonndata(ypred, false, false));
%% plotting it
plot (pred, 'b-');
hold on;
plot(ypred, 'g-');
hold off
legend('Test Data', 'Fitted Values')
%% seasonal 
x_train = tonndata (traindat , false , false) ;
test = tonndata (pred, false, false);
lags= 50;
n_neurons= 15;
net_sbr = narnet(1:lags, n_neurons);
net_sbr.trainFcn = 'trainbr';
net_sbr.divideFcn= 'divideind';
net_sbr.trainParam.epochs = 10000; 
seasonal_lags = 7;
feed = zeros(7 , size((301: seasonal_lags:1000) ,2)) ;
for i = 1:7
feed (i, :) = (300+i):seasonal_lags:1000;
end
feed = feed' ;
train_indices = feed (:) ;
Train_ix = [linspace(1, 300,300)'; train_indices] ;
Val_ix = [];
net_sbr.divideParam = struct( 'trainInd' , Train_ix , ...
                              'valInd' , Val_ix , ...
                               'testInd' , []);
net_sbr.trainParam.showWindow= 1;
[Xs, Xi, Ai, Ts] = preparets( net_sbr, {} , {} , x_train ) ;
net_sbr = train( net_sbr, Xs, Ts, Xi, Ai ) ;
%%
net_sbr = closeloop(train(net_sbr, Xs, Ts, Xi, Ai)); 

%% 
ypred = nan(100+lags, 1);
ypred = tonndata(ypred, false, false);
ypred(1:lags) = x_train(end-(lags-1):end);
[xc, xic, aic, tc] = preparets(net_sbr, {}, {}, ypred);
%converting to predictions
ypred = fromnndata(net_sbr(xc, xic, aic), true, false, false);
residuals = ypred - pred;
output = num2cell(ypred)';

%% performance check
MSE = perform(net_sbr, testpred, output);
MAE = mae(net_sbr, testpred, tonndata(ypred, false, false));
%% plotting it
plot (pred, 'b-');
hold on;
plot(ypred, 'g-');
hold off
legend('Test Data', 'Fitted Values')
%% Alphabet Recognition
appcr1

%% pima diabetes
pima = load('pidstart.mat');
inputs = pima.Xnorm';
targets = hardlim(pima.Y');
%% creating test and training set
rp = randperm(768);
validset = rp(1:115);
testset = rp(116:230);
%% Cross entropy, LM, 5 Neurons
n_neurons = 5
net = patternnet(n_neurons);
net.layers{1}.transferFcn = 'logsig';
net.outputs{1}.transferFcn = 'logsig';
net.divideFcn = 'divideind';
net.trainFcn = 'trainlm';
Train_ix = [setdiff(1:768, [validset testset])];
Val_ix = [validset]
Test_ix = [testset]
net.divideParam = struct('trainInd', Train_ix, ...
                        'valInd', Val_ix, ...
                        'testInd', Test_ix);
[net, tr] = train(net, inputs, targets);
%% Cross entropy, LM, 10 Neurons
n_neurons = 5
net2 = patternnet(n_neurons);
net2.layers{1}.transferFcn = 'logsig';
net2.outputs{1}.transferFcn = 'logsig';
net2.divideFcn = 'divideind';
net2.trainFcn = 'trainlm';
net2.divideParam = struct('trainInd', Train_ix, ...
                        'valInd', Val_ix, ...
                        'testInd', Test_ix);
[net2, tr2] = train(net2, inputs, targets);
%% MSE, BR, 5 Neurons
n_neurons = 5
net_br= patternnet(n_neurons);
net_br.trainFcn = 'trainbr';
net_br.layers{1}.transferFcn = 'logsig';
net_br.outputs{1}.transferFcn = 'logsig';
net_br.divideFcn = 'divideind';
net_br.divideParam = struct('trainInd', Train_ix, ...
                        'valInd', Val_ix, ...
                        'testInd', Test_ix);
[net_br, ~] = train(net_br, inputs, targets);

%% MSE, BR, 10 Neurons
n_neurons = 10
net_br2= patternnet(n_neurons);
net_br2.trainFcn = 'trainbr';
net_br2.layers{1}.transferFcn = 'logsig';
net_br2.outputs{1}.transferFcn = 'logsig';
net_br2.divideFcn = 'divideind';
net_br2.divideParam = struct('trainInd', Train_ix, ...
                        'valInd', Val_ix, ...
                        'testInd', Test_ix);
[net_br2, ~] = train(net_br2, inputs, targets);
%% Cross entropy, GD, 5 Neurons
n_neurons = 5
net_gd = patternnet(n_neurons);
net_gd.trainFcn = 'traingd';
net_gd.layers{1}.transferFcn = 'logsig';
net_gd.outputs{1}.transferFcn = 'logsig';
net_gd.divideFcn = 'divideind';
net_gd.divideParam = struct('trainInd', Train_ix, ...
                        'valInd', Val_ix, ...
                        'testInd', Test_ix);
[net_gd, tr_gd] = train(net_gd, inputs, targets);
%% Cross entropy, GD, 10 Neurons
n_neurons = 10
net_gd2 = patternnet(n_neurons);
net_gd2.trainFcn = 'traingd';
net_gd2.layers{1}.transferFcn = 'logsig';
net_gd2.outputs{1}.transferFcn = 'logsig';
net_gd2.divideFcn = 'divideind';
net_gd2.divideParam = struct('trainInd', Train_ix, ...
                        'valInd', Val_ix, ...
                        'testInd', Test_ix);
[net_gd2, tr_gd] = train(net_gd2, inputs, targets);
%% Cross entropy, SCG, 5 Neurons
n_neurons = 5
net_scg = patternnet(n_neurons);
net_scg.trainFcn = 'trainscg';
net_scg.layers{1}.transferFcn = 'logsig';
net_scg.outputs{1}.transferFcn = 'logsig';
net_scg.divideFcn = 'divideind';
net_scg.divideParam = struct('trainInd', Train_ix, ...
                        'valInd', Val_ix, ...
                        'testInd', Test_ix);
[net_scg, tr_gd] = train(net_scg, inputs, targets);
%% Cross entropy, SCG, 10 Neurons
n_neurons = 10
net_scg2 = patternnet(n_neurons);
net_scg2.trainFcn = 'trainscg';
net_scg2.layers{1}.transferFcn = 'logsig';
net_scg2.outputs{1}.transferFcn = 'logsig';
net_scg2.divideFcn = 'divideind';
net_scg2.divideParam = struct('trainInd', Train_ix, ...
                        'valInd', Val_ix, ...
                        'testInd', Test_ix);
[net_scg2, tr_gd] = train(net_scg2, inputs, targets);
%% MSE, SCG, 10 Neurons
n_neurons = 10
net_scg3 = patternnet(n_neurons);
net_scg3.trainFcn = 'trainscg';
net_scg3.layers{1}.transferFcn = 'logsig';
net_scg3.outputs{1}.transferFcn = 'logsig';
net_scg3.divideFcn = 'divideind';
net_scg3.performFcn = 'mse';
net_scg3.divideParam = struct('trainInd', Train_ix, ...
                        'valInd', Val_ix, ...
                        'testInd', Test_ix);
[net_scg3, tr_gd] = train(net_scg3, inputs, targets);
%% for evaluation
testX = inputs(:, tr.testInd);
testT = targets(:, tr.testInd); 
%% getting the fitted values
test_yhat = net(testX); %LM 5 CE fit
test_yhat2 = net2(testX); %LM 10 CE fit
test_yhat_br = net_br(testX); %BR 5 MSE fit
test_yhat_br2 = net_br2(testX); %BR 10 MSE fit
test_yhat_gd = net_gd(testX); %GD 5 CE fit
test_yhat_gd2 = net_gd2(testX); %GD 10 CE fit
test_yhat_scg = net_scg(testX); %SCG 5 CE fit
test_yhat_scg2 = net_scg2(testX); %SCG 10 CE fit
test_yhat_scg3 = net_scg3(testX); %SCG 10 MSE fit
%% confusion matrices
figure, plotconfusion(testT, test_yhat), legend('LM Cross-Entropy 5'); %LM 5 CE fit
figure, plotconfusion(testT, test_yhat2), legend('LM Cross-Entropy 10'); %LM 10 CE fit
figure, plotconfusion(testT, test_yhat_br), legend('BR MSE 5');
figure, plotconfusion(testT, test_yhat_br2), legend('BR MSE 10');
figure, plotconfusion(testT, test_yhat_gd), legend('GD CE 5');
figure, plotconfusion(testT, test_yhat_gd2), legend('GD CE 10');
figure, plotconfusion(testT, test_yhat_scg), legend('SCG CE 5');
figure, plotconfusion(testT, test_yhat_scg2), legend('SCG CE 10');
figure, plotconfusion(testT, test_yhat_scg3), legend('SCG MSE 10');

%% roc curves
figure, plotroc(testT, test_yhat, 'LM 5', testT, test_yhat2, 'LM 10', testT, test_yhat_br, 'BR 5', testT, test_yhat_br2, 'BR 10'); 
figure, plotroc(testT, test_yhat_gd, 'GD 5', testT, test_yhat_gd2, 'GD 10', testT, test_yhat_scg, 'SCG 5', testT, test_yhat_scg2, 'SCG 10'); 

%%
figure, plotroc(testT, test_yhat_scg3, 'SCG MSE')