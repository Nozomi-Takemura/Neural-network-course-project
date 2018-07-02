clear;
d1=linspace(-5,5,1000);
tr_output=sinc(sqrt(d1.*d1));

input=[d1];
output=[tr_output];
net=fitnet(100,'trainscg');
net.performParam.regularization = 0.000001 ;
[net,tr]=train(net,input,output);

aa = net(input(:,tr.testInd));
(output(:,tr.testInd)' -aa')'*(output(:,tr.testInd)' -aa')/length(aa)
dif2 = sum(abs(output(:,tr.testInd)' -aa'))
tr.time(end)
tr.stop
MSE = perform(net,output(:,tr.testInd),aa)

%{
test_yhat = net(test_input);
(test_yhat' -test_output')'*(test_yhat' -test_output')/length(test_output)
dif = sum(abs(test_yhat' -test_output'))
%}

%test_yhat = net(test_input);
%{
fitsurface = fit(input(:,tr.trainInd)',output(:,tr.trainInd)', 'poly22','Normalize','on');
pred = fitsurface(input(1,tr.testInd)',input(2,tr.testInd)');
mse_poly = (output(:,tr.testInd)' -pred)'*(output(:,tr.testInd)' -pred)/length(pred)
dif2 = sum(abs(output(:,tr.testInd)' -pred))
names5 = coeffnames(fitsurface5);
size(names5)
%}
fitsurface2 = fit(input(:,tr.trainInd)',output(:,tr.trainInd)', 'poly2','Normalize','on');
pred2 = fitsurface2(input(1,tr.trainInd)');
mse_poly = (output(:,tr.trainInd)' -pred2)'*(output(:,tr.trainInd)' -pred2)/length(pred2)
dif2 = sum(abs(output(:,tr.trainInd)' -pred2))
names2 = coeffnames(fitsurface2);
size(names2)


%poly3
fitsurface3 = fit(input(:,tr.trainInd)',output(:,tr.trainInd)', 'poly3','Normalize','on');
pred3 = fitsurface3(input(1,tr.testInd)');
mse_poly3 = (output(:,tr.testInd)' -pred3)'*(output(:,tr.testInd)' -pred3)/length(pred3)
dif3 = sum(abs(output(:,tr.testInd)' -pred3))

fitsurface3 = fit(input(:,tr.trainInd)',output(:,tr.trainInd)', 'poly3','Normalize','on');
pred3 = fitsurface3(input(1,tr.trainInd)');
mse_poly3 = (output(:,tr.trainInd)' -pred3)'*(output(:,tr.trainInd)' -pred3)/length(pred3)
dif3 = sum(abs(output(:,tr.trainInd)' -pred3))
names3 = coeffnames(fitsurface3);
size(names3)
%poly4
%poly4
fitsurface4 = fit(input(:,tr.trainInd)',output(:,tr.trainInd)', 'poly4','Normalize','on');
pred4 = fitsurface4(input(1,tr.testInd)');
mse_poly4 = (output(:,tr.testInd)' -pred4)'*(output(:,tr.testInd)' -pred4)/length(pred4)
dif4 = sum(abs(output(:,tr.testInd)' -pred4))

fitsurface4 = fit(input(:,tr.trainInd)',output(:,tr.trainInd)', 'poly4','Normalize','on');
pred4 = fitsurface4(input(1,tr.trainInd)');
mse_poly4 = (output(:,tr.trainInd)' -pred4)'*(output(:,tr.trainInd)' -pred4)/length(pred4)
dif4 = sum(abs(output(:,tr.trainInd)' -pred4))
names4 = coeffnames(fitsurface4);
size(names4)

fitsurface5 = fit(input(:,tr.trainInd)',output(:,tr.trainInd)', 'poly5','Normalize','on');
pred5 = fitsurface5(input(1,tr.trainInd)');
mse_poly5 = (output(:,tr.trainInd)' -pred5)'*(output(:,tr.trainInd)' -pred5)/length(pred5)
dif5 = sum(abs(output(:,tr.trainInd)' -pred5))
names5 = coeffnames(fitsurface5);
size(names5)


fitsurface9 = fit(input(:,tr.trainInd)',output(:,tr.trainInd)', 'poly9','Normalize','on');
pred9 = fitsurface9(input(1,tr.trainInd)');
mse_poly9 = (output(:,tr.trainInd)' -pred9)'*(output(:,tr.trainInd)' -pred9)/length(pred9)
dif9 = sum(abs(output(:,tr.trainInd)' -pred9))
names9 = coeffnames(fitsurface9);
size(names9)

%more than 5?
p = polyfit(input(:,tr.trainInd),output(:,tr.trainInd),100)



size(sfit(fitsurface,input(:,tr.testInd)'))

plot3(test_input(1,:),test_input(2,:),test_output,'b*');
title('Mexican Hat Function')
xlabel('Input - First Dimension')
ylabel('Input - Second Dimension')
zlabel('Output')
hold on;
plot3(test_input(1,:),test_input(2,:),test_yhat,'ro');
legend({'Generated Output','Estimated Output'});
hold off;
display(tr.best_perf)
%%
MSE = perform(net,test_d1,test_yhat)
MSE = perform(net,test_output, test_yhat)