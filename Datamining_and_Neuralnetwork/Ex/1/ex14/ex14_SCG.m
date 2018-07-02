%% 

%Ex1
cd 'H:/Documents/package/'
%cd /home/nozomi/package/
addpath(pwd)
nnd11gn

%% 
%Ex2
    %2

x = linspace(0,1, 21);
y = -cos(0.8*pi*x);
net = fitnet(2);
net = configure(net,x,y);
net.inputs{1}.processFcns= {};
net.outputs{2}.processFcns = {};
[net, tr] = train(net, x, y);

[biases,weights]=hidden_layer_weights(net);
[fun] = hidden_layer_transfer_function(net);
[biases_o,weights_o]=output_layer_weights(net);
[fun_o] = output_layer_transfer_function(net);
%biases(i) and weights(i)   contain the bias and
%weight for the ith hidden neuro
%% 

x1 = zeros(1,21);
x2 = zeros(1,21);
out_p = zeros(1,21);
for i=1:21
    x1(1,i) = fun(weights(1)*x(i)+biases(1));
    x2(1,i) = fun(weights(2)*x(i)+biases(2));
    out_p(1,i) = fun_o(weights_o(1)*x1(1,i) + weights_o(2)*x2(1,i) + biases_o);
end

%%
clf
hold on
plot(x,y, 'LineWidth',2)   
plot(x, x1, '--','LineWidth',2)
plot(x, x2, '-.','LineWidth',2)
plot(x, out_p, ':','LineWidth',2)
leg1=legend('$y_{p}$','$x^{1}_{p}$','$x^{2}_{p}$','$output_{p}$','Location','Best');
set(leg1,   'Interpreter','latex');
set(leg1,'FontSize',14);
leg2 = xlabel('$x_{p}$');
set(leg2,'Interpreter','latex');
set(leg2,'FontSize',14);

%%
%7.   The  learnt  weights,  bias  and  transfer  function  of  the
%output  neuron  define a  function  together.  Use this function to 
%deneoutputp, the output of the network for each patternp . plot outputp 
%against the input patterns and compare this to the plot in (3).    

sum1_o = dot(out1, weights_o(1,:))
%creat col vecotr of new input to output layer
x_o = [out1,out2];
sum_o = x_o * weights_o'
sum_o = sum_o' + biases_o;
%output = tanh(sum_o)
out_o = fun_o(sum_o);

%%
    %8
% set seeds
clear;
rng(0408);
matrix = zeros(3*3*4*4, 9);
seq = cell(3*3*4*4,1);
%matrix = zeros(3*3*2*2, 7);

%count_i = 0;
%count_j = 0;
%count_k = 0;
count_p = 0;
%%Try these   ['trainrp', 'trainscg','trainbfg','trainlm'] for fitnet
% Try disp(simple_net.trainParam.min_grad), simple_net.trainParam.max_fail]
% = [1.0000e-07(def), 600(def*100)] or [1.0000e-10(def/10^3), 6(def)]
%Try (simple_net.performParam.regularization) = [0(def),linspace(0,1,0.1) ] 
%cd H:\Documents\DMNN\fig1_3\
cd C:\Temp\CG2
%cd C:\Temp\BP
tic;
for i = [30, 150, 1200]
    %count_i = count_i + 1;
    train_x = linspace(-1,1,i);
    val_x = linspace(-0.9,0.9,i);
    for j = [0.2, 0.6, 1.2]
        %count_j = count_j + 1;
        train_y = cos(2*pi*train_x) + j*randn(size(train_x));
        val_y=cos(2*pi*val_x)+j*randn(size(val_x));
        x = [train_x val_x];
        y = [train_y val_y];
            for k = [1,3,5,7]
            %for k = 1:2
                %count_k = count_k + 1;
                net=fitnet(k,'trainscg');
                net.divideFcn='divideind';
                net.divideParam=struct('trainInd',1:i,'valInd',i+1:i*2,'testInd',[]);
                for p=linspace(0,1,4)
                %for p=linspace(0,1,2)   
                    count_p = count_p + 1;
                    jj = round(j*10);
                    pp = round(p*10);
                    net.performParam.regularization = p;
                    %notestset
                    [net,tr] = train(net,x,y);
                    train_yhat=net(train_x);
                    g = figure; set(g, 'Visible', 'off');
                    plotperform(tr);
                    %saveas(g,sprintf('ValPerf_QN_data%i_noise%i_%ihuni_%ireg.eps',i, jj, k, pp)); % will create FIG1, FIG2,...
                    %saveas(g,sprintf('ValPerf_LM_data%i_noise%i_%ihuni_%ireg.eps',i, jj, k, pp), 'epsc'); % will create FIG1, FIG2,...
                    %saveas(g,sprintf('ValPerf_CG_data%i_noise%i_%ihuni_%ireg.eps',i, jj, k, pp)); % will create FIG1, FIG2,...
                    %saveas(g,sprintf('ValPerf_BG_data%i_noise%i_%ihuni_%ireg.eps',i, jj, k, pp)); % will create FIG1, FIG2,...
                    saveas(g,sprintf('ValPerf_SCG_data%i_noise%i_%ihuni_%ireg.eps',i, jj, k, pp),'epsc'); % will create FIG1, FIG2,...
                    close(g)
                    %pause;
                    h=figure; set(h, 'Visible', 'off');
                    plot(train_x,train_y,'r*');
                    hold on;
                    plot(train_x,train_yhat,'-');
                    plot(train_x,cos(2*pi*train_x),'g-');
                    hold off;
                    legend('Training_Set','Approximated_Function','True_Function');
                    %title(['Training setting: Backpropagation with ' 'the number of traing data=' num2str(i) ', amount of noise=' num2str(j) ', number of hidden units=' num2str(k) ' and regulatization term=' num2str(p)]);
                    title(['Training setting: scaled conjugate gradient with ' 'the number of traing data=' num2str(i) ', amount of noise=' num2str(j) ', number of hidden units=' num2str(k) ' and regulatization term=' num2str(p)]);
                    %saveas(h,sprintf('QN_data%i_noise%i_%ihuni_%ireg.eps',i, jj, k, pp)); % will create FIG1, FIG2,...
                    %saveas(h, sprintf('LM_data%i_noise%i_%ihuni_%ireg.eps',i, jj, k, pp),'epsc');
                    saveas(h,sprintf('SCG_data%i_noise%i_%ihuni_%ireg.eps',i, jj, k, pp),'epsc'); % will create FIG1, FIG2,...
                    %saveas(h,sprintf('BP_data%i_noise%i_%ihuni_%ireg.eps',i, jj, k, pp)); % will create FIG1, FIG2,...
                    %each time have to caluculate!!3*3*10*11
                    %u = (count_i-1)*(3*10*11)+(count_j-1)*(10*11)+(count_k-1)*11+count_p;
                    %u = (count_i - 1)*(3*2*2) + (count_j - 1)*(2*2) + (count_k - 1)*2 + count_p
                    u = count_p;
                    matrix(u,1) = i;
                    matrix(u,2) = j;
                    matrix(u,3) = k;
                    matrix(u,4) = p;
                    if strcmp(tr.stop,'Minimum gradient reached.')
                        matrix(u,5) = 0;
                    elseif strcmp(tr.stop, 'Validation stop.')
                        matrix(u,5) = 1;
                    else
                        matrix(u,5) = 2;
                    end
                    seq{u} = tr.stop;
                    disp(matrix(u,5))
                    tr.stop
                    %matrix(u,6) = net.IW{1}(1);
                    %qq(1,i) = net.IW{1}(2);
                    matrix(u,6) = tr.best_perf;
                    matrix(u,7) = tr.best_vperf;
                    matrix(u,8) = tr.best_epoch;
                    matrix(u,9) = tr.time(length(tr.epoch));
                end 
            end
    end
end
t = toc;
save('C:\Temp\CG2\SCGstat.mat', 'matrix')
save('C:\Temp\CG2\SCGstat.mat', 't')
save('C:\Temp\CG2\SCGstat.mat', 'seq')
save('C:\Temp\CG2\all.mat')
%for o=1:3
%    h=figure
%    plot([1:5],[1:5])
%    cd H:\Documents\DMNN\fig1_3\
%    saveas(h,sprintf('BP_%dTdata_%dnoise_%dhuni.eps',i, j, k)); % will create FIG1, FIG2,...
%    title(['Training setting: resilient backpropagation with ' 'the number of traing data ' num2str(i) ', amount of noise ' num2str(j) ', number of hidden units ' num2str(k)]);
%end                
linspace(-1,1,)

%% 
%case; m = 1
clear;
clf;
rng(0408);
j = 0.2;
count_k = 0;
matrix = [];
cd C:\Temp\fig14\SCG_dim1;
sec = cell(4*4,1)
tic;
for m = [30,150,450,1200]
%train_x = linspace(-5,5,11);
%train_y = sinc(sqrt(train_x.^2)) + j*randn(size(train_x))
%test
%val_x = linspace(-4.9,4.9,11)
%val_y = sinc(sqrt(val_x.^2)) + j*randn(size(val_x));
    x = linspace(-5,5,m);
    y = sinc(sqrt(x.^2)) + j*randn(size(x));
    for k = [1,3,5,7]
        count_k = count_k + 1;
        net=fitnet(1,'trainscg');
        [net,tr] = train(net,x,y);
        train_yhat=net(x(tr.trainInd));
        val_yhat=net(x(tr.valInd));
        test_yhat = net(x(tr.testInd));
        %{
        h=figure; set(h, 'Visible', 'off');
        plot(x(tr.trainInd),y(tr.trainInd),'r*');
        hold on;
        plot(x(tr.trainInd),train_yhat,'-');
        plot(x(tr.trainInd),sinc(sqrt(x(tr.trainInd).^2)),'g-');
        hold off;
        legend('Training_Set','Approximated_Function','True_Function');
        title(['Training setting: scaled conjugate gradient with ' 'the number of traing data=' num2str(m) ', amount of noise = 0.1, number of hidden units=' num2str(k)]);
        saveas(h,sprintf('SCG_data%i_noise01_%ihuni_%ireg.eps',m, k, pp),'epsc'); % will create FIG1, FIG2,...
        g=figure; set(g, 'Visible', 'off');
        %}
        %{      
        plot(x(tr.testInd),y(tr.testInd),'r*');
        hold on;
        plot(x(tr.testInd),test_yhat,'-');
        plot(x(tr.testInd),sinc(sqrt(x(tr.testInd).^2)),'g-');
        hold off;
        plot(sinc(sqrt(x(tr.testInd).^2)),test_yhat,'r*');
        %}
        %hold on;
        mat_tra = [sinc(sqrt(x(tr.trainInd).^2))', train_yhat'];
        %plot(sinc(sqrt(x(tr.trainInd).^2)),train_yhat,'r*');
        h = figure; set(h, 'Visible', 'off');
        plot(mat_tra(:,1), mat_tra(:,1), 'r*');
        hold on;
        mat_test = [sinc(sqrt(x(tr.testInd).^2))', test_yhat'];
     %plot(sinc(sqrt(x(tr.trainInd).^2)),train_yhat,'r*');
        plot(mat_test(:,1), mat_test(:,1), 'bo');
        hold off;
        legend('Training_set', 'Test_set');
        title(['Approxmiated function against sinc(r): scaled conjugate gradient with the number of traing data=' num2str(m) ', amount of noise = 0.1, number of hidden units=' num2str(k) ', dimension = 1']);
        saveas(h,sprintf('SCG_data%i_noise01_huni_%i_yhat_vs_sinc.eps',m, k),'epsc'); % will create FIG1, FIG2,...
        %hold off;
        u = count_k;
        matrix(u,1) = m;
        %matrix(u,2) = j;
        matrix(u,2) = k;
        %matrix(u,4) = p;
        if strcmp(tr.stop,'Minimum gradient reached.')
            matrix(u,3) = 0;
        elseif strcmp(tr.stop, 'Validation stop.')
            matrix(u,3) = 1;
        else
            matrix(u,3) = 2;
        end
        seq{u} = tr.stop;
        disp(matrix(u,3));
        tr.stop
        %matrix(u,6) = net.IW{1}(1);
        %qq(1,i) = net.IW{1}(2);
        matrix(u,4) = tr.best_perf;
        matrix(u,5) = tr.best_vperf;
        matrix(u,6) = tr.best_epoch;
        matrix(u,7) = tr.time(length(tr.epoch));
    %train_ind = tr.trainInd;
%val_ind = tr.valInd;
%test_ind = tr.testInd;
    end
end
t = toc;
save('C:\Temp\fig14\SCG_dim1\stat.mat', 'matrix')
save('C:\Temp\fig14\SCG_dim1\stat.mat', 't')
save('C:\Temp\fig14\SCG_dim1\stat.mat', 'seq')
save('C:\Temp\fig14\SCG_dim1\all.mat')

x = [train_x val_x];
y = [train_y val_y];
for k = [1,3,5,7]
%for k = 1:2
%count_k = count_k + 1;
net=fitnet(k,'trainscg');
net.divideFcn='divideind';
net.divideParam=struct('trainInd',1:i,'valInd',i+1:i*2,'testInd',[]);
for p=linspace(0,1,4)
%for p=linspace(0,1,2)   
count_p = count_p + 1;
jj = round(j*10);
pp = round(p*10);
net.performParam.regularization = p;
%notestset
[net,tr] = train(net,x,y);
train_yhat=net(train_x);













%% 

ab_p = (J'*J)\(J'*y_p);
a_p = ab_p(1) ;
b_p = ab_p(2) ;
y0_p = a_p*0 + b_p ;
y1_p = a_p*1 + b_p ;
plot([0,1],[y0_p,y1_p])
legend('Measurements','L2 fit.','L2 fit. pert.');


tx = linspace(-1,1,100);
ty = linspace(-1,1,100);
vx = linspace(-0.9,0.9,100);
vy = linspace(-0.9,0.9,100);
X = [tx vx];
Y = [ty vy];
simple_net = fitnet(2, 'trainlm')
simple_net.divideFcn = 'divideind';
simple_net.divideParam = struct('trainInd',1:((length(x))/2),'valInd',((length(x))/2)+1:length(x),'testInd',[]);



pp = zeros(1,10);
qq = zeros(1,10);
oo = zeros(1,10);
for i = 1:10;
    simple_net = fitnet(2, 'trainlm')
    simple_net.divideFcn = 'divideind';
    simple_net.divideParam = struct('trainInd',1:(length(tx)),'valInd',length(tx))+1:2*length(tx),'testInd',[]);
    [simple_net, tr] = train(simple_net, tx, ty);
    pp(1,i) = simple_net.IW{1}(1);
    qq(1,i) = simple_net.IW{1}(2);
    oo(1,i) = tr.best_vperf    
end
[X,Y] = meshgrid(pp,qq);
Z = oo;
surf(X,Y,Z)
xlabel('Initial weight for 1st neuron');
ylabel('Initial weight for 2st neuron');
zlabel('lowest cost for validation set');

plotperform()
plottrainstate()
ploterrhist()
plotregression()
plotfit()

%lay1-1st neu
sum1 = dot(x ,weights(1,:)) + biases(1);
%lay1-2nd neu
sum2 = dot(x ,weights(2,:)) + biases(2);
%1lay-out from neu1
%out1 = tanh(sum1);
out1 = fun(sum1);
%1lay-out from neu2
%out2 = tanh(sum2);
out2 = fun(sum2);
%outlay-1st
sum1_o = dot(out1, weights_o(1,:))
%creat col vecotr of new input to output layer
x_o = [out1,out2];
sum_o = x_o * weights_o'
sum_o = sum_o' + biases_o;
%output = tanh(sum_o)
out_o = fun_o(sum_o);
input.data = [out1; out2];
latexTable(input)

%%  
for p=1:21


%%

    %fprintf('%d \n',i)
 %inppat(:,i) = linspace(0,1, 21)'

%x = inppat(:,i)';
x = linspace(0,1, 21)';
y = -cos(0.8*pi*x);
net = fitnet(2);
net = configure(net,x,y);
net.inputs{1}.processFcns= {};
net.outputs{2}.processFcns = {};
[net, tr] = train(net, x, y);
[biases,weights]=hidden_layer_weights(net);
[fun] = hidden_layer_transfer_function(net);
sum1 = dot(x ,weights(1,:)) + biases(1);
%lay1-2nd neu
sum2 = dot(x ,weights(2,:)) + biases(2);
%1lay-out from neu1
%out1 = tanh(sum1);
out1 = fun(sum1);
%1lay-out from neu2
%out2 = tanh(sum2);
out2 = fun(sum2);

x = linspace(0, 1, 5)';

    %3
%y = -cos(0.8*pi*x.');
y = -cos(0.8*pi*x);
plot(x, y)
hold on;
xlabel('x')
ylabel('y')
    %4
net = fitnet(2);
net = configure(net,x,y);
net.inputs{1}.processFcns= {};
net.outputs{2}.processFcns = {};
[net, tr] = train(net, x, y);

% The neurons in the hidden layer
%use a hyperbolic tangent (tanh) transfer function, whereas those in the output layer use a linear
%function.




%---------------------------
 %Plot the values ofyp,x1pandx2pagainstp .  How would you model the relationship between ypand (x1p;x2p )?
 [x,t] = simplefit_dataset;
net = feedforwardnet(2); view(net)
net = configure(net,x,t); view(net)
net.inputs{1}.processFcns= {};
net.outputs{2}.processFcns = {};
[net, tr] = train(net, x, t);
[biases,weights]=hidden_layer_weights(net);

%% 
