%% Train NAR Network and Predict on New Data

%%
% Load the simple time-series prediction data and create a NAR network.
T = simplenar_dataset; 
net = narnet(1:2,10); 

%%
% Prepare the time series data using |preparets| and train the network.
[Xs,Xi,Ai,Ts] = preparets(net,{},{},T);
net = train(net,Xs,Ts,Xi,Ai);
view(net)

%%
% Calculate the network performance.
[Y,Xf,Af] = net(Xs,Xi,Ai); 
perf = perform(net,Ts,Y) 

%% 
% To predict the output for the next 20 time steps, first simulate the
% network in closed loop form.
[netc,Xic,Aic] = closeloop(net,Xf,Af); 
view(netc)

%%
% The network only has one input. In closed loop mode, this input is
% joined to the output.

%%
% To simulate the network 20 time steps ahead, input an empty cell array of
% length 20. The network requires only the initial conditions given in
% |Xic| and |Aic|.
y2 = netc(cell(0,20),Xic,Aic)
