cd package/;
addpath(pwd);
appcr1

%%
clear;
%[x,t] = iris_dataset;
net = patternnet(10);
net = configure(net,X,Y);
iniw = net.IW;
iniLW = net.LW;
iniw{1} = ones(10,30)
[net tr] = train(net,X,Y);
view(net)
t = net(X);
perf = perform(net,Y,t);
classes = vec2ind(t);
[tpr,fpr,thresholds] = roc(Y,t);