
clear all;

fprintf('Support Vector Classification\n')
fprintf('_____________________________\n')


% Making X and Y
% %*******************
if 1
    N=150;
    randn('state',0);
% train set
    Xtr=randn(2,N);
    Xtr(:,1:N/2)=Xtr(:,1:N/2)+2;
    Xtr(:,N/2+1:N)=Xtr(:,N/2+1:N)-2;
    Ytr=[ones(1,N/2),-1*ones(1,N/2)];
    datatr=[Xtr;Ytr];
    datatr=datatr(:,randperm(N));
    Xtr=datatr(1:2,:); X=Xtr;
    Ytr=datatr(3,:)'; Y=Ytr;
% test set
    Xtest=randn(2,N);
    Xtest(:,1:N/2)=Xtest(:,1:N/2)+2;
    Xtest(:,N/2+1:N)=Xtest(:,N/2+1:N)-2;
    Ytest=[ones(1,N/2),-1*ones(1,N/2)]';
end


% Dimension of the input:
%************************
[d N]=size(X);

% visualisation of the input data:
%*********************************
% figure;
% gplotmatrix(X',X',Y)
% title('Scatterplot of the input')


% Parameters
%************
% Type of kernels: linear, polynomial, gaussian_rbf, ..
type='gaussian_rbf';
fprintf('Type of kernel %s \n',type)

epsilon=1e-10;
% Penalty factor for missclassification:
C=10;

% For a polynomiale kernel:
dp=2;
% gaussian or exponential  RBF kernel:
sigma=1;

% For a MLP kernel:
scale=0;
offset=0;
% For a B-spline kernel
ds=7;


% parameters:
%************
switch type
case 'linear'
    par=[];
    type_number=0;
case 'polynomial'
    % the order of the polynomial is given by par
    fprintf('the order is: %d \n',dp)
    par=[dp];
    type_number=1;
case 'gaussian_rbf'
    % the value of sigma is given by par
    fprintf('sigma is:  %d \n',sigma)
    type_number=2;
    par=[sigma];
    
case 'exponential_rbf'  
    % the value of sigma is given by par
    fprintf('sigma is:  %d \n',sigma)
    par=[sigma];
    type_number=3;
case 'mlp'
    % w and the bias of tanh is given by par
    par=[scale offset];
    type_number=4;
case 'Bspline'
    % the order of the B-spline is given by par
    par=[ds];
    type_number=6;
case 'Local RBF' %'local gaussian_rbf' 
    % the value of sigma is given by par
    fprintf('sigma is:  %d \n',sigma)
    type_number=7;    
    par=[sigma];
end



% training the QP-svm:
% *********************
t=cputime;

solution=SVM_TRAINQP(X,Y,epsilon,type_number,C,par);

time=cputime-t;
fprintf('The training needed %d',time),fprintf(' seconds.\n');

alpha1=solution{1,1};
b1=solution{1,2};


% Plotting training set.
%****************************
if size(X,1)==2
    figure; subplot(1,2,1), SVMplot(X,Y,X,Y,alpha1,b1,40,type_number,par,epsilon);
    title('training set');
end


% Classification of the training set:
%************************************
Yclass = sign(fsvmclass(X,Y,type_number,epsilon,alpha1,b1,X,par)); % classify on X
misclass = length(find((Yclass+Y)==0));
fprintf('number of misclassified points of the training set: %d (%3.1f%%) \n',misclass,100*misclass/length(Y'));


% Classification of the test set:
%**********************************
if ~(isempty(Xtest))
    if size(X,1)==2
        subplot(1,2,2),SVMplot(Xtest,Ytest,X,Y,alpha1,b1,40,type_number,par,epsilon);
        title('test set');
    end
    
    latent=fsvmclass(X,Y,type_number,epsilon,alpha1,b1,Xtest,par); % classify on Xtest
    Yclass=sign(latent);
    misclass=length(find((Yclass+Ytest)==0));
    fprintf('misclassified points of the test set: %d (%3.1f%%) \n',misclass,100*misclass/length(Ytest'));
end
