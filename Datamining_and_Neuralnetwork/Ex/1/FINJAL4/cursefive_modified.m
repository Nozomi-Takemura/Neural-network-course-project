clear;
%Data Generation
d1=linspace(-5,5,10);
d2=linspace(-5,5,10);
d3=linspace(-5,5,10);
d4=linspace(-5,5,10);
d5=linspace(-5,5,10);
[gen1,gen2,gen3,gen4,gen5]=ndgrid(d1,d2,d3,d4,d5);

tr_input=([gen1(:), gen2(:), gen3(:), gen4(:), gen5(:) ].');
intermed=tr_input.*tr_input;
tr_output=sinc(sqrt((intermed(1,:)+intermed(2,:)+intermed(3,:)+intermed(4,:)+intermed(5,:))));
 
val_d1=linspace(-4.5,4.5,4);
val_d2=linspace(-4.5,4.5,4);
val_d3=linspace(-4.5,4.5,4);
val_d4=linspace(-4.5,4.5,4);
val_d5=linspace(-4.5,4.5,4);
[val_gen1,val_gen2,val_gen3,val_gen4,val_gen5]=ndgrid(val_d1,val_d2,val_d3,val_d4,val_d5);
val_input=([val_gen1(:), val_gen2(:), val_gen3(:), val_gen4(:), val_gen5(:) ].');
val_intermed=val_input.*val_input;
val_output=sinc(sqrt((val_intermed(1,:)+val_intermed(2,:)+val_intermed(3,:)+val_intermed(4,:)+val_intermed(5,:))));
input=[tr_input val_input];
output=[tr_output val_output];
% 
test_d1=linspace(-4.75,4.75,4);
test_d2=linspace(-4.75,4.75,4);
test_d3=linspace(-4.75,4.75,4);
test_d4=linspace(-4.75,4.75,4);
test_d5=linspace(-4.75,4.75,4);
[test_gen1,test_gen2,test_gen3,test_gen4,test_gen5]=ndgrid(test_d1,test_d2,test_d3,test_d4,test_d5);

test_input=([test_gen1(:), test_gen2(:), test_gen3(:) , test_gen4(:) , test_gen5(:)  ].');
test_intermed=test_input.*test_input;
test_output=sinc(sqrt((test_intermed(1,:)+test_intermed(2,:)+test_intermed(3,:)+test_intermed(4,:)+test_intermed(5,:))));
 net=fitnet(,'trainscg');%net=fitnet(50,'trainscg');
 net.divideFcn='dividerand';
% net.divideParam=struct('trainInd',[1:2500],...
%          'valInd',[2501:2900],...
% % %         'testInd',[]);%notestset
 net.performParam.regularization = 0.000001 ;
[net,tr]=train(net,input,output);
% 
test_yhat = net(test_input);
(test_yhat' -test_output')'*(test_yhat' -test_output')/length(test_output)
dif = sum(abs(test_yhat' -test_output'))


aa = net(input(:,tr.testInd))
(output(:,tr.testInd)' -aa')'*(output(:,tr.testInd)' -aa')/length(aa);
dif2 = sum(abs(output(:,tr.testInd)' -aa'))

tr.time(end)
tr.stop

% 
plot3(test_input(1,:),test_input(2,:),test_output,'b*');
title('Five Dimensional Sinc Function - First two')
 xlabel('Input - First Dimension')
ylabel('Input - Second Dimension')
zlabel('Output')
hold on;
plot3(test_input(1,:),test_input(2,:),test_yhat,'ro');
 legend({'Generated Output','Estimated Output'});
hold off;

figure
plot3(test_input(3,:),test_input(4,:),test_output,'b*');
title('Five Dimensional Sinc Function - Third and Fourth')
 xlabel('Input - Third Dimension')
ylabel('Input - Fourth Dimension')
zlabel('Output')
hold on;
plot3(test_input(3,:),test_input(4,:),test_yhat,'ro');
 legend({'Generated Output','Estimated Output'});
hold off;


figure
plot(test_input(5,:),test_output,'b*');
title('Five Dimensional Sinc Function - Fifth')
xlabel('Input - Fifth Dimension')
ylabel('Output')
hold on;
plot(test_input(5,:),test_yhat,'ro');
 legend({'Generated Output','Estimated Output'});
hold off;