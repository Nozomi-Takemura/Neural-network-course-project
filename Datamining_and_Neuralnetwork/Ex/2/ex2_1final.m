clear all

cd /home/nozomi;
traindat = importdata('/home/nozomi/Documents/2017courses/DMNN/Ex/Data/lasertrain.dat');
pred = importdata('/home/nozomi/Documents/2017courses/DMNN/Ex/Data/laserpred.dat');
cd 'Documents/2017courses/DMNN/Ex/2/';
addpath(pwd);
plot(1:length(traindat), traindat, '-k'), hold on;
plot((length(traindat)+1):(length(traindat)+length(pred)), pred, ' -r');
xlabel('Time')
ylabel('Intensity')
legend('Training Data', 'Test Data')
hold off
%% training and validation
x_train = tonndata(traindat, false, false);
testpred = tonndata(pred, false, false);

%%  test nozomi-123117
lags = 70;
n_neurons = 30;
net = narnet(1:lags, n_neurons);
view(net);
%net.trainFcn = 'trainlm';
net.trainFcn = 'trainbr';
net.performParam.regularization = 1e-6;
net.divideParam.testRatio = 0;
net.divideParam.trainRatio = 0.85;
net.divideParam.valRatio = 0.15;
[Xs, Xi, Ai, Ts] = preparets(net, {}, {}, x_train);
net = train(net,Xs,Ts,Xi,Ai);
[Y,Xf,Af] = net(Xs,Xi,Ai); 
perf = perform(net,Ts,Y)

[netc,Xic,Aic] = closeloop(net,Xf,Af); 
view(netc)
y2 = netc(cell(0,100),Xic,Aic)
A = cell2mat(y2);
plot (pred, 'b-');
hold on;
plot(A', 'r-');
hold off
legend('Test Data', 'Fitted Values')
MAE =  sum(abs(A'-pred))/length(pred);
MSE = (A'-pred)'*(A'-pred);
MSE = MSE/length(pred)


plot(1:length(traindat), traindat, '-k'), hold on;
plot((length(traindat)+1):(length(traindat)+length(pred)), pred, ' -r');
plot((length(traindat)+1):(length(traindat)+length(pred)), A', ' -b');
xlabel('Time')
ylabel('Intensity')
legend('Training Data', 'Test Data')
hold off