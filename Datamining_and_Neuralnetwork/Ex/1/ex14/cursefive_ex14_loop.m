%%
cd 'H:/Documents/package/'
%cd /home/nozomi/package/
addpath(pwd)
clear;
clf;
rng(0408);
%%
j = 0.2;
count_k = 0;
matrix_sum = [];
cd C:\Temp\1221ex14;
sec = cell(4*4,1);


for m = [30,150,450,1200]
%Data Generation
d1=linspace(-5,5,10);
d2=linspace(-5,5,10);
d3=linspace(-5,5,10);
d4=linspace(-5,5,10);
d5=linspace(-5,5,10);
[gen1,gen2,gen3,gen4,gen5]=ndgrid(d1,d2,d3,d4,d5);

tr_input=([gen1(:), gen2(:), gen3(:), gen4(:), gen5(:) ].')
intermed=tr_input.*tr_input;
tr_output=sinc(sqrt((intermed(1,:)+intermed(2,:)+intermed(3,:)+intermed(4,:)+intermed(5,:))));

tr_output_noise = tr_output  + j*randn(size(tr_output));
%{
input=[tr_input val_input];
output=[tr_output val_output];
output_noise = [tr_output_noise val_output_noise];
%}

% 
test_d1=linspace(-4.75,4.75,4);
test_d2=linspace(-4.75,4.75,4);
test_d3=linspace(-4.75,4.75,4);
test_d4=linspace(-4.75,4.75,4);
test_d5=linspace(-4.75,4.75,4);
[test_gen1,test_gen2,test_gen3,test_gen4,test_gen5]=ndgrid(test_d1,test_d2,test_d3,test_d4,test_d5);

test_input = ([test_gen1(:), test_gen2(:), test_gen3(:) , test_gen4(:) , test_gen5(:)  ].')
test_intermed=test_input.*test_input;
test_output=sinc(sqrt((test_intermed(1,:)+test_intermed(2,:)+test_intermed(3,:)+test_intermed(4,:)+test_intermed(5,:))));

net=fitnet(200,'trainscg');
net.divideFcn='dividerand';
input=[tr_input];
output=[tr_output_noise];


net.performParam.regularization = 0.000001 ;
[net,tr]=train(net,input,output);
test_input = tr_input(:,tr.testInd);
test_intermed=test_input.*test_input;
test_output=sinc(sqrt((test_intermed(1,:)+test_intermed(2,:)+test_intermed(3,:)+test_intermed(4,:)+test_intermed(5,:))));
% 
test_yhat = net(test_input);
%

plot3(test_input(1,:),test_input(2,:),test_output,'b*');
title('5 dimensional sinc function: 1st and 2nd dimension')
 xlabel('Input value of 1st dimension')
ylabel('Input value of 2nd dimension')
zlabel('Output')
hold on;
plot3(test_input(1,:),test_input(2,:),test_yhat,'ro');
 legend({'sinc(x)','Approximated function'});
hold off;
saveas(h,sprintf('BP_data%i_noise01_huni_%i_reg000001_yhat_vs_sinc.eps',m, k),'epsc'); % will create FIG1, FIG2,...
        %hold off;

figure
plot3(test_input(3,:),test_input(4,:),test_output,'b*');
title('5 dimensional sinc function: 3rd and 4th dimenstion')
 xlabel('Input - 3rd dimension')
ylabel('Input - 4th dimension')
zlabel('Output')
hold on;
plot3(test_input(3,:),test_input(4,:),test_yhat,'ro');
 legend({'sinc(x)','Approximated function'});
hold off;


figure
plot(test_input(5,:),test_output,'b*');
title('5 dimensional sinc function - 5th')
xlabel('Input - 5th dimension')
ylabel('Output')
hold on;
plot(test_input(5,:),test_yhat,'ro');
 legend({'sine(x)','Approximated function'});
hold off;       

%u = count_k;
u - 1;
matrix_sum(u,1) = m;
%matrix(u,2) = j;
matrix_sum(u,2) = k;
%matrix(u,4) = p;
if strcmp(tr.stop,'Minimum gradient reached.')
    matrix_sum(u,3) = 0;
elseif strcmp(tr.stop, 'Validation stop.')
    matrix_sum(u,3) = 1;
else
    matrix_sum(u,3) = 2;
end
seq{u} = tr.stop;
disp(matrix_sum(u,3));
tr.stop
%matrix(u,6) = net.IW{1}(1);
%qq(1,i) = net.IW{1}(2);
matrix_sum(u,4) = tr.best_perf;
matrix_sum(u,5) = tr.best_vperf;
matrix_sum(u,6) = tr.best_epoch;
matrix_sum(u,7) = tr.time(length(tr.epoch));
%train_ind = tr.trainInd;
%val_ind = tr.valInd;
%test_ind = tr.testInd;
    end
end
t = toc;
save('C:\Temp\fig14\BP_dim5\stat.mat', 'matrix_sum')
save('C:\Temp\fig14\BP_dim5\stat.mat', 't')
save('C:\Temp\fig14\BP_dim5\stat.mat', 'seq')
save('C:\Temp\fig14\BP_dim5\all.mat')