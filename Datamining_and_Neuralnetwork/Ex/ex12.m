%Ex1
%cd 'H:/Documents/Documents/package/'
cd /home/nozomi/package/
addpath(pwd)
nnd11gn
%%
%Ex2
    %2
%fprintf('%d \n',i)
 %inppat(:,i) = linspace(0,1, 21)'

%x = inppat(:,i)';
clear;
mat = zeros(3,21)
%out2 = zeros(1,21)
%for i=1:21
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
%lay1-1st neu
sum1 = sum(x*weights(1)) + biases(1);
%lay1-2nd neu
sum2 = sum(x*weights(2)) + biases(2);
%1lay-out from neu1
%out1 = tanh(sum1);
out1 = fun(sum1);
%1lay-out from neu2
%out2 = tanh(sum2);
out2 = fun(sum2);
%mat(1,i) = i;
%mat(2,i) = out1;
%mat(3,i) = out2;
%outlay-1st
sum1_o = out1*weights_o(1) + out2*weights_o(2) + biases_o;
out_o = fun_o(sum1_o);
%end

h = figure; set(h, 'Visible', 'off');
plot(mat(1,:), mat(3,:), 'r-');
plot(mat(1,:), y);
plot([1:21], fun(x*weights(1)+biases(1)), 'b-')
hold on;
plot([1:21], fun(x*weights(2)+biases(2)), 'r-');
plot([1:21], y, 'g-');
plot([1:21], fun_o(fun(x*weights(1)+biases(1))*weights_o(1) + fun(x*weights(2)+biases(2))*weights_o(2) + biases_o), 'b*')
%[Xtr,Ytr] = meshgrid(inp_yh(1,:),inp_yh(2,:));
%[Xte,Yte] = meshgrid(inp_tyh(1,:),inp_tyh(2,:));
 %plot(sinc(sqrt(x(tr.trainInd).^2)),train_yhat,'r*');
plot(fun(weights(1)*x+biases(1)), 'b-');
plot(fun(weights(2)*x+biases(2)), 'r-');
plot(y, 'g-')
hold off;
%leg1 = legend('$x_{p}^{1}$', '$x_{p}^{2}$', 'Location', 'Best');
leg1 = legend('$x_{p}^{1}$', '$x_{p}^{2}$', '$y_{p}$', 'Location', 'Best');
%l1 = xlabel('$x_{p}$');
l1 = xlabel('p');
leg1.FontSize = 14;
l1.FontSize = 14;
set(leg1,'Interpreter','latex');
set(l1,'Interpreter','latex');
%ylabel('')
%title('Activaitons from the hidden layer');
t1 = title('$x_{p}^{1}, x_{p}^{2}, y_{p}$ vs p');
set(t1, 'Interpreter', 'latex')
saveas(h,sprintf('LM_data%i_noise01_huni_%i_yhat_vs_sinc.eps',m, k),'epsc'); % will create FIG1, FIG2,...
%hold off;

plot(x,y, 'b*')
hold on;
plot(x, fun_o(weights_o(1)*fun(weights(1)*x+biases(1)) + weights_o(2)*fun(weights(2)*x+biases(2)) + biases_o), 'r-')
plot(x,y, 'b*')
leg1 = legend('$-cos(0.8 \pi x_{p})$', '$output_{p}$', 'Location', 'Best');
l1 = xlabel('$x_{p}$');
leg1.FontSize = 14;
l1.FontSize = 14;
set(leg1,'Interpreter','latex');
set(l1,'Interpreter','latex');
%ylabel('')
title('Output of the network');











%creat col vecotr of new input to output layer
x_o = [out1,out2];
sum_o = x_o * weight_o'
sum_o = sum_o' + biases_o;
%output = tanh(sum_o)
out_o = fun_o(sum_o);
indics = [out]
input.data = [out1: out2];
latexTable(input)
for i=1:2
    plot(x())
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

        [biases,weights]=hidden_layer_weights(net);
        [fun] = hidden_layer_transfer_function(net);
        [biases_o,weights_o]=output_layer_weights(net);
        [fun_o] = output_layer_transfer_function(net);
        %biases(i) and weights(i)   contain the bias and
        %weight for the ith hidden neuro

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
        sum1_o = dot(out1, weight_o(1,:))
        %creat col vecotr of new input to output layer
        x_o = [out1,out2];
        sum_o = x_o * weight_o'
        sum_o = sum_o' + biases_o;
        %output = tanh(sum_o)
        out_o = fun_o(sum_o);
        indics = [out]
        input.data = [out1: out2];
        latexTable(input)
        for i=1:2
            plot(x())
