function sol=classification_dataset(N,type)
% This function generate different datasets. 
% N=number of trainingpoints
% type, is teh type of datastes you want:
% chessboard, gaussian, spiral, adult, Australian credit (acr), Bupa Liver disorder (bupa), 
% German credit (german), Stalog Haert disease (haert), John Hopkings University Ionosphere (iono),
% Pima Indian Diabetes (pima), Sonar (sonar), Tic-Tac-Toe (ttt), Wisconsin Breast Cancer (wbc),
% Iris (iris1v23, iris2v13, iris3v12)

X=[];
Y=[];
Xtest=[];
Ytest=[];
gam=[];
sigma=[];

switch type
case 'chessboard'
    N1=ceil(N/16);
    for i=1:4
        for j=1:4
            X1=rand(2,N1);
            X1(1,:)=X1(1,:)+i;
            X1(2,:)=X1(2,:)+j;
            Y1=(-1)^(i+j)*ones(N1,1);
            X=[X X1];
            Y=[Y;Y1];
        end
    end
    
    % randomize:
    p=randperm(N);
    X=X(:,p);
    Y=Y(p);
    X=X(:,1:N);
    Y=Y(1:N);
    
    Ninput=ceil(2*N/3);
    Xtest=X(:,Ninput+1:N);
    Ytest=Y(Ninput+1:N);
    X=X(:,1:Ninput);
    Y=Y(1:Ninput);
    
    sigma=1;
    gam=10;
    
case 'gaussian'
    X=7*rand(2,N);
    X(1,1:N/2)=X(1,1:N/2)+5;
    X(2,1:N/2)=X(2,1:N/2)+5;
    Y=ones(N,1);
    Y(N/2+1:N)=Y(N/2+1:N)-2;
    
    % randomize:
    p=randperm(N);
    X=X(:,p);
    Y=Y(p);
    X=X(:,1:N);
    Y=Y(1:N);
    
    Ninput=ceil(2*N/3);
    Xtest=X(:,Ninput+1:N);
    Ytest=Y(Ninput+1:N);
    X=X(:,1:Ninput);
    Y=Y(1:Ninput);
    
    gam=1;
    sigma=50;
    
case 'adult'
    load adultb.mat
    X=Xtrval(:,1:N);
    Y=Ytrval(1:N);
    [d N]=size(X);
    gam=gammarbf*N/1000;
    sigma=sigmarbf/sqrt(d);
    
case 'acr'
    load acrb.mat
    X=Xtrval;
    Y=Ytrval;
    [d N]=size(X);
    gam=gammarbf*N/1000;
    sigma=sigmarbf/sqrt(d); 
    
case 'bupa'
    load bupab.mat
    X=Xtrval;
    Y=Ytrval;
    [d N]=size(X);
    gam=gammarbf*N/1000;
    sigma=sigmarbf/sqrt(d); 
    
case 'german'
    load germanb.mat
    X=Xtrval;
    Y=Ytrval;
    [d N]=size(X);
    gam=gammarbf*N/1000;
    sigma=sigmarbf/sqrt(d);
    
case 'haert'
    load haertb.mat
    X=Xtrval;
    Y=Ytrval;
    [d N]=size(X);
    gam=gammarbf*N/1000;
    sigma=sigmarbf/sqrt(d); 
    
case 'iono'
    load ionob.mat
    X=Xtrval;
    Y=Ytrval;
    [d N]=size(X);
    gam=gammarbf*N/1000;
    sigma=sigmarbf/sqrt(d); 
    
case 'wisb'
    load wisbb.mat
    X=Xtrval;
    Y=Ytrval;
    [d N]=size(X);
    gam=gammarbf*N/1000;
    sigma=sigmarbf/sqrt(d);
    
case 'pima'
    load pimab.mat
    X=Xtrval;
    Y=Ytrval;
    [d N]=size(X);
    gam=gammarbf*N/1000;
    sigma=sigmarbf/sqrt(d); 
    
case 'sonar'
    load sonarb.mat
    X=Xtrval;
    Y=Ytrval;
    [d N]=size(X);
    gam=gammarbf*N/1000;
    sigma=sigmarbf/sqrt(d); 
    
case 'ttt'
    load tttb.mat
    X=Xtrval;
    Y=Ytrval;
    [d N]=size(X);
    gam=gammarbf*N/1000;
    sigma=sigmarbf/sqrt(d); 
    
case 'iris1v23'
    load iris1v23.mat
    X=X';
    N=length(Y);
    p=randperm(N);
    X=X(:,p);
    Y=Y(p);
    X=X(:,1:N);
    Y=Y(1:N);
    
    Ninput=ceil(2*N/3);
    Xtest=X(:,Ninput+1:N);
    Ytest=Y(Ninput+1:N);
    X=X(:,1:Ninput);
    Y=Y(1:Ninput);
    [d N]=size(X);
    
    gam=[10];
    sigma=[2]; 
    
case 'iris2v13'
    load iris2v13.mat
    X=X';
    N=length(Y);
    p=randperm(N);
    X=X(:,p);
    Y=Y(p);
    X=X(:,1:N);
    Y=Y(1:N);
    
    Ninput=ceil(2*N/3);
    Xtest=X(:,Ninput+1:N);
    Ytest=Y(Ninput+1:N);
    X=X(:,1:Ninput);
    Y=Y(1:Ninput);
    [d N]=size(X);
    
    gam=[10];
    sigma=[2]; 
    
case 'iris3v12'
    load iris3v12.mat
    X=X';
    N=length(Y);
    p=randperm(N);
    X=X(:,p);
    Y=Y(p);
    X=X(:,1:N);
    Y=Y(1:N);
    
    Ninput=ceil(2*N/3);
    Xtest=X(:,Ninput+1:N);
    Ytest=Y(Ninput+1:N);
    X=X(:,1:Ninput);
    Y=Y(1:Ninput);
    [d N]=size(X);
    
    gam=[10];
    sigma=[2]; 
end



% Output:
sol=cell(1,6);

sol{1,1}=X;
sol{1,2}=Y;
sol{1,3}=Xtest;
sol{1,4}=Ytest;
sol{1,5}=gam;
sol{1,6}=sigma;
