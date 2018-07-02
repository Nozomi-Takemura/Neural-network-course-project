clear;
%Data Generation
%{
d1=linspace(-5,5,80);
d2=linspace(-5,5,80);
[gen1,gen2]=meshgrid(d1,d2);
%Z=sinc(sqrt(X.*X+Y.*Y));
tr_input=([gen1(:), gen2(:) ].')
intermed=tr_input.*tr_input;%elementwise multiple
tr_output=sinc(sqrt((intermed(1,:)+intermed(2,:))));

val_d1=linspace(-4.5,4.5,40);
val_d2=linspace(-4.5,4.5,40);
[val_gen1,val_gen2]=meshgrid(val_d1,val_d2);
val_input=([val_gen1(:), val_gen2(:) ].')
val_intermed=val_input.*val_input;
val_output=sinc(sqrt((val_intermed(1,:)+val_intermed(2,:))));
input=[tr_input val_input];
output=[tr_output val_output];
%}

d1=linspace(-5,5,50);
d2=linspace(-5,5,50);
[gen1,gen2]=meshgrid(d1,d2);
%Z=sinc(sqrt(X.*X+Y.*Y));
tr_input=([gen1(:), gen2(:) ].');
intermed=tr_input.*tr_input;%elementwise multiple
tr_output=sinc(sqrt((intermed(1,:)+intermed(2,:))));

input=[tr_input];
output=[tr_output];

test_d1=linspace(-4.75,4.75,40);
test_d2=linspace(-4.75,4.75,40);
[test_gen1,test_gen2]=meshgrid(test_d1,test_d2);
%Z=sinc(sqrt(X.*X+Y.*Y));
test_input=([test_gen1(:), test_gen2(:) ].')
test_intermed=test_input.*test_input;
test_output=sinc(sqrt((test_intermed(1,:)+test_intermed(2,:))));

%net=fitnet(5,'trainbfg');
net=fitnet(50,'trainbfg');
net=fitnet(1000,'trainscg');
% net.divideFcn='divideind';
% net.divideParam=struct('trainInd',[1:2500],...
%         'valInd',[2501:2900],...
%         'testInd',[]);%notestset
net.performParam.regularization = 0.000001 ;
[net,tr]=train(net,input,output);
%{
test_yhat = net(test_input);
(test_yhat' -test_output')'*(test_yhat' -test_output')/length(test_output)
dif = sum(abs(test_yhat' -test_output'))
%}

aa = net(input(:,tr.testInd));
(output(:,tr.testInd)' -aa')'*(output(:,tr.testInd)' -aa')/length(aa)
dif2 = sum(abs(output(:,tr.testInd)' -aa'))
tr.time(end)
tr.stop
%test_yhat = net(test_input);
fitsurface = fit(input(:,tr.trainInd)',output(:,tr.trainInd)', 'poly22','Normalize','on');
pred = fitsurface(input(1,tr.testInd)',input(2,tr.testInd)');
mse_poly = (output(:,tr.testInd)' -pred)'*(output(:,tr.testInd)' -pred)/length(pred)
dif2 = sum(abs(output(:,tr.testInd)' -pred))
names5 = coeffnames(fitsurface5);
size(names5)

fitsurface2 = fit(input(:,tr.trainInd)',output(:,tr.trainInd)', 'poly22','Normalize','on');
pred2 = fitsurface2(input(1,tr.trainInd)',input(2,tr.trainInd)');
mse_poly = (output(:,tr.trainInd)' -pred2)'*(output(:,tr.trainInd)' -pred2)/length(pred2)
dif2 = sum(abs(output(:,tr.trainInd)' -pred2))
names2 = coeffnames(fitsurface2);
size(names2)


%poly3
fitsurface3 = fit(input(:,tr.trainInd)',output(:,tr.trainInd)', 'poly33','Normalize','on');
pred3 = fitsurface3(input(1,tr.testInd)',input(2,tr.testInd)');
mse_poly3 = (output(:,tr.testInd)' -pred3)'*(output(:,tr.testInd)' -pred3)/length(pred3)
dif3 = sum(abs(output(:,tr.testInd)' -pred3))

fitsurface3 = fit(input(:,tr.trainInd)',output(:,tr.trainInd)', 'poly33','Normalize','on');
pred3 = fitsurface3(input(1,tr.trainInd)',input(2,tr.trainInd)');
mse_poly3 = (output(:,tr.trainInd)' -pred3)'*(output(:,tr.trainInd)' -pred3)/length(pred3)
dif3 = sum(abs(output(:,tr.trainInd)' -pred3))
names3 = coeffnames(fitsurface3);
size(names3)
%poly4
%poly4
fitsurface4 = fit(input(:,tr.trainInd)',output(:,tr.trainInd)', 'poly44','Normalize','on');
pred4 = fitsurface4(input(1,tr.testInd)',input(2,tr.testInd)');
mse_poly4 = (output(:,tr.testInd)' -pred4)'*(output(:,tr.testInd)' -pred4)/length(pred4)
dif4 = sum(abs(output(:,tr.testInd)' -pred4))

fitsurface4 = fit(input(:,tr.trainInd)',output(:,tr.trainInd)', 'poly44','Normalize','on');
pred4 = fitsurface4(input(1,tr.trainInd)',input(2,tr.trainInd)');
mse_poly4 = (output(:,tr.trainInd)' -pred4)'*(output(:,tr.trainInd)' -pred4)/length(pred4)
dif4 = sum(abs(output(:,tr.trainInd)' -pred4))
disp(fit)

fitsurface5 = fit(input(:,tr.trainInd)',output(:,tr.trainInd)', 'poly55','Normalize','on');
pred5 = fitsurface5(input(1,tr.trainInd)',input(2,tr.trainInd)');
mse_poly5 = (output(:,tr.trainInd)' -pred5)'*(output(:,tr.trainInd)' -pred5)/length(pred5)
dif5 = sum(abs(output(:,tr.trainInd)' -pred5))
names5 = coeffnames(fitsurface5);
size(names5)

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