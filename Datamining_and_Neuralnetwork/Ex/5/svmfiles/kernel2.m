
function K=kernel(X,Y,type_nummer,varargin)
% par heeft een betekennis afhankelijk het type kernel.
% als Y leeg is wordt de kernel matrix berekend.
% type polynomial: par is de dimenie van de parameter
% type gaussian rbf: par = sigma en moet als derde argument worden ingegeven
% type exponetial rbf: par=sigma en moet als derde argument worden ingegeven
% type mlp : par1 = scale, par 2 = offset en moet als vierde argument worden ingegeven
%  sp: optie voor spaarse matrices te gebruiken.
sp=0;

[m1 n1]=size(X);

m2=m1;
n2=n1;


switch nargin
case 4
    par1=varargin{1};
case 5
    par1=varargin{1};
    par2=varargin{2};
end

equal=isempty(Y);

%parameter for local rbf
 cutoff=3;


if equal
    if sp % if we use the sparse formulation
        K= sparse([],[],[],n1,n2,0)
    else
        K=zeros(n1,n2);
    end
    
    switch type_nummer
    case 0 % linear 
        %     for i=1:n1
        %         for j=1:n2
        %             K(j,i)=X(:,i)'*Y(:,j); 
        %         end
        %     end
        K=X'*X;;
    case 1 % Polynomial
        if nargin<4
            fprintf('WARNING:there are not enough inputparameters')
        end
        d=par1;
        %     for i=1:n1
        %         for j=1:n2
        %             K(j,i)=(X(:,i)'*Y(:,j)+1)^d; 
        %         end
        %     end
        K=(X'*X+ones(n1,n2)).^d;
        
        
    case 2 %'gaussian_rbf'  
        if nargin<4
            fprintf('WARNING:there are not enough inputparameters')
        end
        sigma=par1;
        for i=1:n1
            K(i,i)=1; 
            for j=1+i:n2
                K(i,j)=exp(-norm(X(:,i)-X(:,j))^2/(sigma^2)); 
                K(j,i)=K(i,j);
            end
        end
        
        
    case 3 %'exponential_rbf'
        if nargin<4
            fprintf('WARNING:there are not enough inputparameters')
        end
        sigma=par1;
        for i=1:n1
            K(i,i)=1; 
            for j=1+i:n2
                K(i,j)=exp(-norm(X(:,i)-X(:,j))/(2*sigma^2)); 
                K(j,i)=K(i,j);
            end
        end
        
    case 4 %'mlp'
        if nargin<5
            fprintf('WARNING:there are not enough inputparameters')
        end
        scale=par1;
        offset=par2;
        for i=1:n1
            K(i,i)=tanh(scale*(tranpose(X(:,i))*X(:,i))-offset);; 
            for j=1:n2
                K(i,j)=tanh(scale*(tranpose(X(:,i))*X(:,j))-offset);
                K(j,i)=K(i,j);
            end
        end
    case 5 %'epanechnikov'
        if nargin<4
            fprintf('WARNING:there are not enough inputparameters')
        end
        h=par1;
        u=norm(X-X)/h;
        if u<=1
            a=0.75*(1-u^2);  
        else
            a=0;
        end
    case 6 %'B-spline'
        if nargin<4
            fprintf('WARNING:there are not enough inputparameters')
        end
        n=par1;
        d=length(X);
        for i=1:n1
            K(i,j)=prod(Bspline(2*n+1,zeros(1,n1)));
            for j=1+i:n2
                a=X(:,i)-X(:,j); 
                K(i,j)=prod(Bspline(2*n+1,a'));
                K(j,i)=K(i,j);
            end
        end
    case 7 %'locale gaussian_rbf'  
        if nargin<4
            fprintf('WARNING:there are not enough inputparameters')
        end
        sigma=par1;
        % localiteits factor:
        theta=cutoff*sigma;
        
        
        if mod(length(X(:,1)),2)==0
            nu=length(X(:,1))+1;
        else
            nu=length(X(:,1));
        end
        
        for i=1:n1
            K(i,i)=1;
            for j=1+i:n2
                
                h=norm(X(:,i)-X(:,j));
                if h>theta
                    K(i,j)=0;
                else
                    K(i,j)=((1-h/theta)^nu)*exp(-h^2/(sigma^2));
                end
                K(j,i)=K(i,j);
            end
        end
        
    end 
        
    else
        [m2 n2]=size(Y);
        if sp % if we use the sparse formulation
            K= sparse([],[],[],n1,n2,0)
        else
            K=zeros(n1,n2);
        end
        
        switch type_nummer
        case 0 % linear 
            %     for i=1:n1
            %         for j=1:n2
            %             K(j,i)=X(:,i)'*Y(:,j); 
            %         end
            %     end
            K=X'*Y;;
        case 1 % Polynomial
            if nargin<4
                fprintf('WARNING:there are not enough inputparameters')
            end
            d=par1;
            %     for i=1:n1
            %         for j=1:n2
            %             K(j,i)=(X(:,i)'*Y(:,j)+1)^d; 
            %         end
            %     end
            K=(X'*Y+ones(n1,n2)).^d;
            
            
        case 2 %'gaussian_rbf'  
            if nargin<4
                fprintf('WARNING:there are not enough inputparameters')
            end
            sigma=par1;
            for i=1:n1
                for j=1:n2
                    K(i,j)=exp(-norm(X(:,i)-Y(:,j))^2/(sigma^2)); 
                end
            end
            
            
        case 3 %'exponential_rbf'
            if nargin<4
                fprintf('WARNING:there are not enough inputparameters')
            end
            sigma=par1;
            for i=1:n1
                for j=1:n2
                    K(i,j)=exp(-norm(X(:,i)-Y(:,j))/(2*sigma^2)); 
                end
            end
            
        case 4 %'mlp'
            if nargin<5
                fprintf('WARNING:there are not enough inputparameters')
            end
            scale=par1;
            offset=par2;
            for i=1:n1
                for j=1:n2
                    K(i,j)=tanh(scale*(tranpose(X(:,i))*Y(:,j))-offset);
                end
            end
        case 5 %'epanechnikov'
            if nargin<4
                fprintf('WARNING:there are not enough inputparameters')
            end
            h=par1;
            u=norm(X-Y)/h;
            if u<=1
                a=0.75*(1-u^2);  
            else
                a=0;
            end
        case 6 %'B-spline'
            if nargin<4
                fprintf('WARNING:there are not enough inputparameters')
            end
            n=par1;
            d=length(X);
            for i=1:n1
                for j=1:n2
                    a=X(:,i)-Y(:,j); 
                    K(i,j)=prod(Bspline(2*n+1,a'));
                end
            end
        case 7 %'locale gaussian_rbf'  
            if nargin<4
                fprintf('WARNING:there are not enough inputparameters')
            end
            sigma=par1;
            % localiteits factor:
            theta=cutoff*sigma;
            
            
            if mod(length(X(:,1)),2)==0
                nu=length(X(:,1))+1;
            else
                nu=length(X(:,1));
            end
            
            for i=1:n1
                for j=1:n2
                    
                    h=norm(X(:,i)-Y(:,j));
                    if h>theta
                        K(i,j)=0;
                    else
                        K(i,j)=((1-h/theta)^nu)*exp(-h^2/(sigma^2));
                    end
                end
            end

        end
    end 
%     K=sparse(K);